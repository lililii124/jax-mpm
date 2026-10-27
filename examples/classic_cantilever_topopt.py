"""3D SIMP sizing of a classical cantilever with a static MPM energy solve.

The elastic energy is integrated at material quadrature points. Quadratic
B-spline shape functions and gradients are the same reference-grid stencil
used by ``mpm_jax.total_lagrangian``. A matrix-free CG equilibrium solve and
``jax.lax.custom_linear_solve`` provide implicit equilibrium sensitivities.
"""
from __future__ import annotations

import argparse
import csv
from collections import deque
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from mpm_jax.core import Config, stencil
from mpm_jax.materials import linear_elastic, linear_stress


E0 = 70.0e9
NU = 0.30
RHO_BASE = 2700.0
RHO_FLOOR = 1.0e-3
VF = 0.38
FORCE_N = 100.0
LENGTH = 0.75
HEIGHT = 0.25
THICKNESS = 0.1875
X0 = 0.125
Y0 = 0.375
Z0 = 0.40625
FILTER_RADIUS_CELLS = 2.0


def geometry(grid: int):
    """Beam cells embedded in a unit MPM background grid."""
    h = 1.0 / grid
    nx, ny, nz = round(LENGTH / h), round(HEIGHT / h), round(THICKNESS / h)
    ix0, iy0, iz0 = round(X0 / h), round(Y0 / h), round(Z0 / h)
    assert abs(nx*h-LENGTH) < 1e-12 and abs(ny*h-HEIGHT) < 1e-12 and abs(nz*h-THICKNESS) < 1e-12
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    cell = np.stack((ix, iy, iz), axis=-1).reshape(-1, 3)
    cell_id = np.arange(cell.shape[0], dtype=np.int32)
    # Eight tensor-product Gauss points per voxel (two per axis).
    g = 1.0 / (2.0 * math.sqrt(3.0))
    signs = np.asarray([(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=np.float64)
    local = 0.5 + g * signs
    ijk = cell[:, None, :] + local[None, :, :]
    xyz = np.empty_like(ijk)
    xyz[..., 0] = X0 + ijk[..., 0] * h
    xyz[..., 1] = Y0 + ijk[..., 1] * h
    xyz[..., 2] = Z0 + ijk[..., 2] * h
    points = xyz.reshape(-1, 3)
    pcell = np.repeat(cell_id, 8)
    cell_volume = h**3
    point_volume = np.full(points.shape[0], cell_volume / 8.0)
    cfg = Config(n_grid=grid, dt=1e-4, domain_size=1.0, boundary_cells=0, material="linear", dim=3)
    support = stencil(jnp.asarray(points), cfg)
    # Explicit non-design clamp pad and finite central load pad.
    clamp_cell = cell[:, 0] < 2
    pad_x = cell[:, 0] >= nx-2
    yc = (cell[:, 1] + 0.5) * h + Y0
    zc = (cell[:, 2] + 0.5) * h + Z0
    load_cell = pad_x & (np.abs(yc-(Y0+HEIGHT/2)) <= 0.0625+h*1e-6) & (np.abs(zc-(Z0+THICKNESS/2)) <= 0.0416667+h*1e-6)
    fixed_cell = clamp_cell | load_cell
    design_cell = ~fixed_cell
    node_axis = np.arange(grid+1) * h
    nxn, nyn, nzn = np.meshgrid(node_axis, node_axis, node_axis, indexing="ij")
    node_xyz = np.stack((nxn, nyn, nzn), axis=-1).reshape(-1, 3)
    fixed_node = node_xyz[:, 0] <= X0 + 1e-12
    # MPM consistent load transfer from a 2x2 Gauss rule over the finite pad.
    p_y = np.asarray([-1, 1]) / math.sqrt(3.0)
    face_points = []
    face_weights = []
    pad_y, pad_z = 0.125, 1.0/12.0
    for a in p_y:
        for b in p_y:
            face_points.append((X0+LENGTH, Y0+HEIGHT/2+0.5*pad_y*a, Z0+THICKNESS/2+0.5*pad_z*b))
            face_weights.append(0.25)
    load_support = stencil(jnp.asarray(face_points), cfg)
    load = np.zeros((node_xyz.shape[0], 3), dtype=np.float64)
    force_weights = np.asarray(load_support.weights) * np.asarray(face_weights)[:, None]
    np.add.at(load[:, 1], np.asarray(load_support.indices).ravel(), (-FORCE_N*force_weights).ravel())
    load[fixed_node] = 0.0
    load_nodes = np.flatnonzero(np.linalg.norm(load, axis=1) > 0)
    assert abs(load[:, 1].sum()+FORCE_N) < 1e-8 and np.max(np.abs(load[:, 0])) < 1e-10
    assert not np.any(fixed_node[load_nodes])
    assert bool(np.all(np.asarray(support.valid)))
    return {
        "grid": grid, "h": h, "shape": (nx, ny, nz), "cell": cell,
        "cell_id_per_point": pcell, "points": points, "point_volume": point_volume,
        "indices": np.asarray(support.indices), "weights": np.asarray(support.weights),
        "gradients": np.asarray(support.gradients), "node_xyz": node_xyz,
        "fixed_node": fixed_node, "load": load, "load_nodes": load_nodes,
        "clamp_cell": clamp_cell, "load_cell": load_cell, "fixed_cell": fixed_cell,
        "design_cell": design_cell,
    }


def gaussian_kernel(radius: float):
    r = max(1, int(math.ceil(radius*2)))
    ax = np.arange(-r, r+1, dtype=np.float64)
    w = np.exp(-0.5*(ax/max(radius, 1e-6))**2)
    k = np.einsum("i,j,k->ijk", w, w, w)
    return k / k.sum()


def build_solver(mesh, beta: float, vf: float, maxiter: int, tol: float, filter_radius_cells: float = FILTER_RADIUS_CELLS):
    ngrid = mesh["grid"]
    shape = mesh["shape"]
    nc = int(np.prod(shape))
    nn = mesh["node_xyz"].shape[0]
    point_cell = jnp.asarray(mesh["cell_id_per_point"])
    vol = jnp.asarray(mesh["point_volume"])
    conn = jnp.asarray(mesh["indices"], dtype=jnp.int32)
    grad = jnp.asarray(mesh["gradients"])
    free = jnp.asarray(~mesh["fixed_node"])
    rhs = jnp.asarray(mesh["load"])
    design = jnp.asarray(mesh["design_cell"].reshape(shape))
    fixed = ~design
    kernel = jnp.asarray(gaussian_kernel(filter_radius_cells))
    ker = kernel[None, None, ...]
    lam = E0*NU/((1+NU)*(1-2*NU))
    mu = E0/(2*(1+NU))
    linear_material=linear_elastic(E0,NU)
    eps_identity = jnp.eye(3, dtype=jnp.float64)
    vf_total = jnp.asarray(vf, dtype=jnp.float64)
    total_cells = float(nc)
    beta_j = jnp.asarray(beta, dtype=jnp.float64)
    def rho_map(q):
        q3 = q.reshape((1, 1) + shape)
        num = lax.conv_general_dilated(q3, ker, (1, 1, 1), "SAME", dimension_numbers=("NCDHW", "OIDHW", "NCDHW"))
        den = lax.conv_general_dilated(jnp.ones_like(q3), ker, (1, 1, 1), "SAME", dimension_numbers=("NCDHW", "OIDHW", "NCDHW"))
        filtered = (num/den).reshape(shape)
        filtered = jnp.where(design, filtered, 1.0)

        def project(x, eta):
            den = jnp.tanh(beta_j*eta)+jnp.tanh(beta_j*(1.0-eta))
            return (jnp.tanh(beta_j*eta)+jnp.tanh(beta_j*(x-eta)))/den

        def volume_residual(eta):
            rho = jnp.where(design, project(filtered, eta), 1.0)
            return jnp.mean(rho)-vf_total

        def bisection(_, state):
            lo, hi = state
            mid = 0.5*(lo+hi)
            go_right = volume_residual(mid) > 0
            return jnp.where(go_right, mid, lo), jnp.where(go_right, hi, mid)

        def root_solve(f, initial):
            lo, hi = lax.fori_loop(0, 48, bisection, (jnp.asarray(0.0), jnp.asarray(1.0)))
            return 0.5*(lo+hi)

        def tangent_solve(g, y):
            return y/g(jnp.asarray(1.0))

        eta = lax.custom_root(volume_residual, jnp.asarray(0.5), root_solve, tangent_solve)
        rho = jnp.where(design, project(filtered, eta), 1.0)
        return jnp.clip(rho.reshape(-1), 1e-8, 1.0)

    def stiffness_action(rho, displacement):
        u = displacement.reshape((nn, 3))
        gu = jnp.einsum("pki,pkj->pij", u[conn], grad)
        strain = 0.5*(gu+jnp.swapaxes(gu, 1, 2))
        sigma = linear_stress(strain,linear_material)
        scale = RHO_FLOOR + (1-RHO_FLOOR)*rho[point_cell]**3
        sigma = sigma*scale[:, None, None]
        fpoint = vol[:, None, None]*jnp.einsum("pij,pkj->pki", sigma, grad)
        out = jnp.zeros((nn, 3), dtype=jnp.float64).at[conn].add(fpoint)
        return out.reshape(-1)

    free_flat = jnp.repeat(free, 3)
    rhs_flat = rhs.reshape(-1)
    rhs_flat = jnp.where(free_flat, rhs_flat, 0.0)
    bnorm = jnp.linalg.norm(rhs_flat)

    def cg(matvec, b, inverse_diagonal):
        x0 = jnp.zeros_like(b)
        r0 = b-matvec(x0)
        z0 = inverse_diagonal*r0
        p0 = z0
        rz0 = jnp.vdot(r0, z0).real
        target = jnp.asarray(tol)*jnp.maximum(jnp.linalg.norm(b), 1.0)

        def cond(state):
            i, _, r, _, _ = state
            return (i < maxiter) & (jnp.linalg.norm(r) > target)

        def body(state):
            i, x, r, p, rz = state
            ap = matvec(p)
            alpha = rz/jnp.maximum(jnp.vdot(p, ap).real, 1e-30)
            xn = x+alpha*p
            rn = r-alpha*ap
            zn = inverse_diagonal*rn
            rzn = jnp.vdot(rn, zn).real
            pn = zn+(rzn/jnp.maximum(rz, 1e-30))*p
            return i+1, xn, rn, pn, rzn

        return lax.while_loop(cond, body, (jnp.asarray(0), x0, r0, p0, rz0))[1]

    def equilibrium(rho):
        scale_cell = RHO_FLOOR + (1-RHO_FLOOR)*rho**3
        scale = scale_cell[point_cell]
        diag_point = vol[:, None]*(lam+2*mu)*scale[:, None]*jnp.sum(grad*grad, axis=2)
        diagonal_node = jnp.zeros((nn,), dtype=jnp.float64).at[conn].add(diag_point)
        inverse_diagonal = 1.0/jnp.maximum(jnp.repeat(diagonal_node,3),1e-20)
        inverse_diagonal = jnp.where(free_flat,inverse_diagonal,1.0)

        def matvec(x):
            x = jnp.where(free_flat, x, 0.0)
            y = stiffness_action(rho, x)
            return jnp.where(free_flat, y, x)

        def solve(mv, b):
            return cg(mv, b, inverse_diagonal)

        u = lax.custom_linear_solve(matvec, rhs_flat, solve=solve, transpose_solve=solve, symmetric=True)
        return u

    equilibrium_jit = jax.jit(equilibrium)

    def responses(q):
        rho = rho_map(q)
        u = equilibrium_jit(rho)
        Ku = stiffness_action(rho, u)
        residual = jnp.linalg.norm(jnp.where(free_flat, Ku-rhs_flat, 0.0))/jnp.maximum(bnorm, 1.0)
        compliance = jnp.vdot(rhs_flat, u).real
        # Recompute integration-point stress and strain for a field audit.
        gu = jnp.einsum("pki,pkj->pij", u.reshape((nn, 3))[conn], grad)
        strain = 0.5*(gu+jnp.swapaxes(gu, 1, 2))
        sigma_solid = linear_stress(strain,linear_material)
        vm = jnp.sqrt(jnp.maximum(0.0, 1.5*jnp.sum((sigma_solid-jnp.trace(sigma_solid, axis1=1, axis2=2)[:, None, None]*eps_identity/3)**2, axis=(1, 2))))
        vf_actual = jnp.mean(rho)
        return compliance, vf_actual, residual, rho, u, jnp.max(vm)

    objective = jax.jit(lambda q: responses(q)[0])
    value_grad = jax.jit(jax.value_and_grad(objective))
    response_jit = jax.jit(responses)
    rho_jit = jax.jit(rho_map)
    action_jit = jax.jit(lambda rho, u: stiffness_action(rho, u))
    return {"responses": response_jit, "value_grad": value_grad, "objective": objective,
            "rho": rho_jit, "action": action_jit, "equilibrium": equilibrium_jit,
            "shape": shape, "ncell": nc,
            "nnodes": nn, "design": np.asarray(mesh["design_cell"]).reshape(shape),
            "free_flat": np.asarray(free_flat), "rhs": np.asarray(rhs_flat),
            "point_cell": np.asarray(mesh["cell_id_per_point"]), "grid": ngrid}


def check_operator(solver, rho):
    n = solver["nnodes"]*3
    rng = np.random.default_rng(1005)
    a, b = rng.normal(size=(2, n))
    free = solver["free_flat"]
    a = np.where(free, a, 0.0)
    b = np.where(free, b, 0.0)
    ka = np.asarray(solver["action"](jnp.asarray(rho), jnp.asarray(a)))
    kb = np.asarray(solver["action"](jnp.asarray(rho), jnp.asarray(b)))
    symmetry = abs(float(np.dot(a, kb)-np.dot(b, ka)))/max(abs(float(np.dot(a,kb))), abs(float(np.dot(b,ka))), 1e-30)
    energy = float(np.dot(a, ka))
    return {"random_vector_symmetry_relative_error": symmetry, "random_vector_energy_aKa": energy,
            "positive_energy": bool(energy > 0)}


def design_checks(rho, shape, fixed_cell, grid):
    field = np.asarray(rho).reshape(shape)
    solid = field >= 0.5
    # Fixed clamp and loading pads are required to remain solid.
    solid |= fixed_cell.reshape(shape)
    labels=np.zeros(shape,dtype=np.int32);count=0
    sx,sy,sz=shape
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                if not solid[i,j,k] or labels[i,j,k]: continue
                count+=1;labels[i,j,k]=count;queue=deque([(i,j,k)])
                while queue:
                    a,b,c=queue.popleft()
                    for ni,nj,nk in ((a-1,b,c),(a+1,b,c),(a,b-1,c),(a,b+1,c),(a,b,c-1),(a,b,c+1)):
                        if 0<=ni<sx and 0<=nj<sy and 0<=nk<sz and solid[ni,nj,nk] and not labels[ni,nj,nk]:
                            labels[ni,nj,nk]=count;queue.append((ni,nj,nk))
    clamp = np.zeros(shape, bool); clamp[:2, :, :] = True
    load = np.zeros(shape, bool); load[-2:, :, :] = fixed_cell.reshape(shape)[-2:, :, :]
    roots = np.unique(labels[clamp]); roots = roots[roots != 0]
    connected = bool(roots.size and np.all(np.isin(labels[load & fixed_cell.reshape(shape)], roots)))
    # Quantify sub-two-cell features with a one-cell spherical opening audit.
    def shifted(a,axis,amount):
        b=np.roll(a,amount,axis=axis)
        sl=[slice(None)]*3;sl[axis]=0 if amount==1 else -1;b[tuple(sl)]=False
        return b
    eroded=solid.copy()
    for axis in range(3): eroded &= shifted(solid,axis,1)&shifted(solid,axis,-1)
    opened=eroded.copy()
    for axis in range(3): opened |= shifted(eroded,axis,1)|shifted(eroded,axis,-1)
    opened |= fixed_cell.reshape(shape)
    lost_fraction = float(np.sum(solid & ~opened)/max(np.sum(solid), 1))
    # Cell-wise cross sections and thresholded volume are reported explicitly.
    # Count complete cross-sectional planes, rather than taking minima over
    # individual voxel lines.  The latter can be zero at the beam perimeter
    # even when every load-bearing plane has a finite section.
    section_x = np.sum(solid, axis=(1, 2))
    section_y = np.sum(solid, axis=(0, 2))
    section_z = np.sum(solid, axis=(0, 1))
    fractions = {
        "threshold_0_5": float(np.mean(solid)),
        "threshold_0_3": float(np.mean(field >= 0.3)),
        "gray_0_1_to_0_9": float(np.mean((field > 0.1) & (field < 0.9))),
        "min_x_section_cells": int(section_x.min()),
        "min_y_section_cells": int(section_y.min()),
        "min_z_section_cells": int(section_z.min()),
        "opening_removed_solid_fraction": lost_fraction,
        "six_connected_component_count": int(count),
        "clamp_to_load_pad_connected": connected,
        "cell_pitch_m": float(1.0/grid),
    }
    return fractions


def save_array(path: Path, **arrays):
    np.savez_compressed(path, **arrays)


def force_balance(mesh, internal_force, enforce=True):
    """Audit the fixed total end load and the clamp reactions."""
    fext=np.asarray(mesh["load"])
    fint=np.asarray(internal_force).reshape(fext.shape)
    fixed=np.asarray(mesh["fixed_node"])
    node=np.asarray(mesh["node_xyz"])
    reaction=(fint-fext)[fixed]
    rf=np.sum(reaction,axis=0)
    rm=np.sum(np.cross(node[fixed],reaction),axis=0)
    applied=np.sum(fext,axis=0)
    am=np.sum(np.cross(node,fext),axis=0)
    ferror=np.linalg.norm(rf+applied)/max(np.linalg.norm(applied),1.0)
    merror=np.linalg.norm(rm+am)/max(np.linalg.norm(am),1.0)
    if enforce and (not np.isfinite(ferror+merror) or ferror>1e-6 or merror>1e-6):
        raise RuntimeError("support force/moment balance exceeded 1e-6")
    return {"applied_resultant_N":applied.tolist(),"applied_moment_about_origin_Nm":am.tolist(),
        "loaded_grid_nodes":int(len(mesh["load_nodes"])),"support_reaction_N":rf.tolist(),
        "support_reaction_moment_about_origin_Nm":rm.tolist(),
        "relative_force_balance_error":float(ferror),"relative_moment_balance_error":float(merror)}


def solve_binary_audit(solver, mesh, rho, out_path):
    """Threshold at 0.5, then independently re-equilibrate the fixed mesh."""
    shape = mesh["shape"]
    field = np.asarray(rho).reshape(shape)
    solid = field >= 0.5
    fixed = mesh["fixed_cell"].reshape(shape)
    solid |= fixed
    binary_rho = np.where(solid, 1.0, RHO_FLOOR).astype(np.float64).ravel()
    u = solver["equilibrium"](jnp.asarray(binary_rho))
    u.block_until_ready()
    Ku = solver["action"](jnp.asarray(binary_rho), u)
    rhs = jnp.asarray(solver["rhs"])
    residual = jnp.linalg.norm(jnp.where(jnp.asarray(solver["free_flat"]), Ku-rhs, 0.0))/jnp.maximum(jnp.linalg.norm(rhs),1.0)
    compliance = jnp.vdot(rhs,u).real
    residual.block_until_ready()
    if not np.isfinite(float(residual)) or float(residual)>1e-6:
        raise RuntimeError("thresholded equilibrium CG residual exceeded 1e-6")
    audit = {"threshold":0.5,"solid_volume_fraction_including_fixed_pads":float(np.mean(solid)),
             "relative_cg_residual":float(residual),"compliance_Nm":float(compliance),
             "force_balance":force_balance(mesh,Ku),
             "topology":design_checks(field,shape,mesh["fixed_cell"],mesh["grid"]),
             "interpretation":"actual second equilibrium solve on thresholded solid/ersatz-void field; the residual modulus remains 0.001 E0"}
    save_array(out_path, rho_binary=binary_rho.reshape(shape), displacement=np.asarray(u).reshape((-1,3)))
    return audit


def solve_true_void_binary(solver, mesh, field, out_path, maxiter=8000, tol=1e-8,
                           checkpoint_interval=500, resume_checkpoint=None):
    """Re-equilibrate the phi>=0.5 solid with exactly zero stiffness in void.

    Grid nodes outside every active particle stencil are eliminated from the
    solve.  This is separate from the SIMP ersatz solve and uses no residual
    void stiffness.
    """
    solid = np.asarray(field).reshape(mesh["shape"]) >= 0.5
    solid |= np.asarray(mesh["fixed_cell"]).reshape(mesh["shape"])
    point_solid = solid.reshape(-1)[np.asarray(mesh["cell_id_per_point"])]
    conn = np.asarray(mesh["indices"], dtype=np.int32)
    nn = len(mesh["node_xyz"])
    supported = np.zeros(nn, dtype=bool)
    supported[conn[point_solid].reshape(-1)] = True
    fixed = np.asarray(mesh["fixed_node"], dtype=bool)
    loaded = np.linalg.norm(mesh["load"], axis=1) > 0.0
    loaded_supported = bool(np.all(supported[loaded]))
    if not loaded_supported:
        return {"status":"invalid_load_on_unsupported_node","active_voxels":int(solid.sum()),
                "active_integration_points":int(point_solid.sum()),
                "unsupported_loaded_nodes":np.flatnonzero(loaded & ~supported).tolist()}
    free = (~fixed) & supported
    free_flat = jnp.repeat(jnp.asarray(free), 3)
    rhs = jnp.asarray(mesh["load"].reshape(-1))
    rhs = jnp.where(free_flat, rhs, 0.0)
    conn_j = jnp.asarray(conn, dtype=jnp.int32)
    grad = jnp.asarray(mesh["gradients"])
    vol = jnp.asarray(mesh["point_volume"])
    active = jnp.asarray(point_solid, dtype=jnp.float64)
    lam = E0*NU/((1+NU)*(1-2*NU))
    mu = E0/(2*(1+NU))
    material = linear_elastic(E0, NU)

    def action(u_flat):
        u = u_flat.reshape((nn, 3))
        gu = jnp.einsum("pki,pkj->pij", u[conn_j], grad)
        strain = 0.5*(gu+jnp.swapaxes(gu, 1, 2))
        sigma = linear_stress(strain, material)*active[:, None, None]
        fpoint = vol[:, None, None]*jnp.einsum("pij,pkj->pki", sigma, grad)
        return jnp.zeros((nn, 3), dtype=jnp.float64).at[conn_j].add(fpoint).reshape(-1)

    # Exact 3x3 nodal diagonal blocks of the integrated isotropic tangent.
    # Block Jacobi handles local vector coupling better than the scalar
    # approximation used by the ersatz optimizer.
    eye3=jnp.eye(3,dtype=jnp.float64)
    g2=jnp.sum(grad*grad,axis=2)
    outer=jnp.einsum("pki,pkj->pkij",grad,grad)
    block_point=vol[:,None,None,None]*active[:,None,None,None]*(
        mu*g2[:,:,None,None]*eye3+(lam+mu)*outer)
    diag_block=jnp.zeros((nn,3,3),dtype=jnp.float64).at[conn_j].add(block_point)
    free_node_j=jnp.asarray(free)
    safe_block=jnp.where(free_node_j[:,None,None],diag_block,eye3)
    inv_block=jnp.linalg.inv(safe_block)

    def precondition(r):
        return jnp.einsum("nij,nj->ni",inv_block,r.reshape((nn,3))).reshape(-1)

    bnorm = jnp.linalg.norm(rhs)
    target = jnp.asarray(tol)*jnp.maximum(bnorm,1.0)
    def mv(x):
        x=jnp.where(free_flat,x,0.0)
        y=action(x)
        return jnp.where(free_flat,y,x)
    if resume_checkpoint:
        saved=np.load(resume_checkpoint)
        if saved["x"].shape!=rhs.shape or saved["residual"].shape!=rhs.shape or saved["direction"].shape!=rhs.shape:
            raise ValueError("zero-void CG checkpoint vector shapes differ from this grid")
        it=int(saved["iteration"]);x=jnp.asarray(saved["x"]);r=jnp.asarray(saved["residual"])
        p=jnp.asarray(saved["direction"]);rz=jnp.asarray(saved["rz"])
        residual_history=np.asarray(saved["residual_history"],dtype=float).tolist()
    else:
        it=0;x=jnp.zeros_like(rhs);r=rhs-mv(x);z=precondition(r);p=z;rz=jnp.vdot(r,z).real
        residual_history=[float(jnp.linalg.norm(r)/jnp.maximum(bnorm,1.0))]

    def run_chunk(iteration,x,r,p,rz,stop_at):
        def cond(state):
            i,_,rr,_,_=state
            return (i<stop_at)&(jnp.linalg.norm(rr)>target)
        def body(state):
            i,xx,rr,pp,rzz=state
            ap=mv(pp);alpha=rzz/jnp.maximum(jnp.vdot(pp,ap).real,1e-30)
            xn=xx+alpha*pp;rn=rr-alpha*ap;zn=precondition(rn);rzn=jnp.vdot(rn,zn).real
            pn=zn+(rzn/jnp.maximum(rzz,1e-30))*pp
            return i+1,xn,rn,pn,rzn
        return lax.while_loop(cond,body,(iteration,x,r,p,rz))

    chunk_jit=jax.jit(run_chunk)
    while it<maxiter and residual_history[-1]>tol:
        it_j,x,r,p,rz=chunk_jit(jnp.asarray(it),x,r,p,rz,jnp.asarray(min(it+checkpoint_interval,maxiter)))
        it=int(it_j)
        x.block_until_ready();rr=float(jnp.linalg.norm(r)/jnp.maximum(bnorm,1.0));residual_history.append(rr)
        save_array(out_path.with_name(out_path.stem+"-checkpoint.npz"),iteration=np.asarray(it),x=np.asarray(x),
                   residual=np.asarray(r),direction=np.asarray(p),rz=np.asarray(rz),
                   residual_history=np.asarray(residual_history,dtype=np.float64))
        progress={"iteration":int(it),"maxiter":int(maxiter),"relative_cg_residual":rr,
                  "target":float(tol),"preconditioner":"exact 3x3 material-point block Jacobi"}
        out_path.with_name(out_path.stem+"-progress.json").write_text(json.dumps(progress,indent=2)+"\n")
    u=x
    internal=action(u)
    residual=jnp.linalg.norm(jnp.where(free_flat,internal-rhs,0.0))/jnp.maximum(bnorm,1.0)
    energy=jnp.vdot(u,internal).real
    residual.block_until_ready();internal_np=np.asarray(internal)
    finite=bool(np.isfinite(np.asarray(u)).all() and np.isfinite(float(residual)) and np.isfinite(float(energy)))
    status="converged" if finite and float(residual)<=tol*1.01 else "not_converged"
    reaction=force_balance(mesh,internal_np,enforce=False)
    rng=np.random.default_rng(4005);a=rng.normal(size=nn*3);b=rng.normal(size=nn*3)
    a=np.where(np.asarray(free_flat),a,0.0);b=np.where(np.asarray(free_flat),b,0.0)
    ka=np.asarray(action(jnp.asarray(a)));kb=np.asarray(action(jnp.asarray(b)))
    symmetry=abs(float(np.dot(a,kb)-np.dot(b,ka)))/max(abs(float(np.dot(a,kb))),abs(float(np.dot(b,ka))),1e-30)
    audit={"status":status,"solver":"matrix-free JAX preconditioned CG; no residual void stiffness",
        "preconditioner":"exact 3x3 block Jacobi from material-point energy diagonal",
        "void_stiffness":0.0,"void_inertia":0.0,"threshold":0.5,
        "active_voxels":int(solid.sum()),"active_integration_points":int(point_solid.sum()),
        "unsupported_nodes_eliminated":int(np.sum(~supported)),"free_supported_nodes":int(np.sum(free)),
        "loaded_nodes":int(np.sum(loaded)),"all_loaded_nodes_supported":loaded_supported,
        "cg_iterations":int(it),"cg_residual_samples_by_chunk":residual_history,
        "relative_cg_residual":float(residual),"compliance_Nm":float(jnp.vdot(rhs,u).real),
        "internal_energy_quadratic_form_J":float(energy),"operator_symmetry_relative_error":symmetry,
        "operator_positive_random_energy":bool(float(jnp.vdot(jnp.asarray(a),jnp.asarray(ka)).real)>0),
        "force_and_reaction_balance":reaction,"threshold_topology":design_checks(solid,mesh["shape"],mesh["fixed_cell"],mesh["grid"])}
    save_array(out_path,rho_binary=solid.astype(np.float32),displacement=np.asarray(u).reshape((-1,3)))
    return audit


def solve_true_void_sparse(mesh, field, out_path, chunk_points=256, diagonal_equilibration=False,
                           iterative=False, cg_tol=1e-10, cg_maxiter=20000,
                           permc_spec="MMD_AT_PLUS_A", weak_support_ratio=0.0):
    """Assemble the same zero-void MPM tangent and solve it with sparse LU.

    This is an independent direct-solver check of the material-point energy;
    only active thresholded-solid quadrature contributes and unsupported nodes
    are removed from the free system.
    """
    import time
    import warnings
    from scipy.sparse import coo_matrix, diags, eye as sparse_eye, kron
    from scipy.sparse.linalg import MatrixRankWarning, cg, norm as sparse_norm, spsolve

    start=time.perf_counter()
    solid=np.asarray(field).reshape(mesh["shape"])>=0.5
    solid|=np.asarray(mesh["fixed_cell"]).reshape(mesh["shape"])
    point_solid=solid.reshape(-1)[np.asarray(mesh["cell_id_per_point"])]
    point_ids=np.flatnonzero(point_solid)
    conn=np.asarray(mesh["indices"],dtype=np.int32)
    grad=np.asarray(mesh["gradients"],dtype=np.float64)
    vol=np.asarray(mesh["point_volume"],dtype=np.float64)
    nn=len(mesh["node_xyz"]);nfull=3*nn
    supported=np.zeros(nn,dtype=bool);supported[conn[point_solid].reshape(-1)]=True
    fixed=np.asarray(mesh["fixed_node"],dtype=bool)
    free=(~fixed)&supported
    loaded=np.linalg.norm(mesh["load"],axis=1)>0.0
    loaded_supported=bool(np.all(supported[loaded]))
    if not loaded_supported:
        return {"status":"invalid_load_on_unsupported_node","solver":"SciPy sparse LU of MPM stiffness",
            "unsupported_loaded_nodes":np.flatnonzero(loaded&~supported).tolist()}
    dofs=np.flatnonzero(np.repeat(free,3))
    node_to_free=np.full(nn,-1,dtype=np.int32);node_to_free[free]=np.arange(int(free.sum()),dtype=np.int32)
    lam=E0*NU/((1+NU)*(1-2*NU));mu=E0/(2*(1+NU));eye=np.eye(3)
    K=None;rows_count=0
    for begin in range(0,len(point_ids),chunk_points):
        ids=point_ids[begin:begin+chunk_points];g=grad[ids]
        ga=g[:,:,:,None,None];gb=g[:,None,None,:,:]
        term_lambda=lam*ga*gb
        dot=np.einsum("pai,pbi->pab",g,g)
        term_mu_diag=mu*dot[:,:,None,:,None]*eye[None,None,: ,None,:]
        term_mu_cross=mu*np.einsum("paj,pbi->paibj",g,g)
        local=(term_lambda+term_mu_diag+term_mu_cross)*vol[ids,None,None,None,None]
        active_nodes=node_to_free[conn[ids]]
        ndof=active_nodes[:,:,None]*3+np.arange(3,dtype=np.int32)[None,None,:]
        shape=(len(ids),27,3,27,3)
        row=np.broadcast_to(ndof[:,:,:,None,None],shape).reshape(-1)
        col=np.broadcast_to(ndof[:,None,None,:,:],shape).reshape(-1)
        data=local.reshape(-1)
        valid=np.broadcast_to(active_nodes[:,:,None,None,None]>=0,shape)&np.broadcast_to(active_nodes[:,None,None,:,None]>=0,shape)
        valid=valid.reshape(-1)
        part=coo_matrix((data[valid],(row[valid],col[valid])),shape=(len(dofs),len(dofs))).tocsr()
        part.sum_duplicates()
        K=part if K is None else K+part
        rows_count+=data.size
        if ((begin//chunk_points)%10)==0 or begin+chunk_points>=len(point_ids):
            progress={"phase":"sparse_assembly","active_points_total":int(len(point_ids)),
                "active_points_assembled":int(min(begin+chunk_points,len(point_ids))),
                "coo_entries_assembled":int(rows_count),"csr_nonzeros_so_far":int(K.nnz),
                "wall_seconds":time.perf_counter()-start}
            out_path.with_name(out_path.stem+"-progress.json").write_text(json.dumps(progress,indent=2)+"\n")
    K=K.tocsr();K.sum_duplicates();Kff=K.tocsc()
    rhs_full=np.asarray(mesh["load"],dtype=np.float64).reshape(-1)
    rhs=rhs_full[dofs]
    rhs_original=rhs.copy()
    Kff_original=Kff
    free_node_ids=np.flatnonzero(free)
    prolongation=None;weak_node=np.zeros(len(free_node_ids),dtype=bool)
    aggregation_info={"weak_support_diagonal_ratio":float(weak_support_ratio),"weak_nodes_aggregated":0,
        "free_nodes_before_aggregation":int(len(free_node_ids)),"free_dofs_before_aggregation":int(len(dofs)),
        "load_nodes_aggregated":0,"affine_row_sum_max_error":0.0,"affine_coordinate_max_error_m":0.0,
        "max_aggregation_weight_abs":0.0,"max_aggregation_neighbor_distance_m":0.0,
        "projected_applied_resultant_error_N":0.0,"projected_applied_moment_error_Nm":0.0}
    if weak_support_ratio>0.0:
        node_diag=np.asarray(Kff.diagonal(),dtype=np.float64).reshape((-1,3)).max(axis=1)
        diag_scale=float(np.median(node_diag[node_diag>0.0]))
        weak_node=node_diag < weak_support_ratio*diag_scale
        loaded_free=np.flatnonzero(loaded[free_node_ids])
        if np.any(weak_node[loaded_free]):
            return {"status":"invalid_aggregation_would_change_loaded_nodes","weak_support_diagonal_ratio":float(weak_support_ratio),
                "loaded_nodes_marked_weak":int(np.sum(weak_node[loaded_free])),"all_original_loads_retained":False}
        strong_pos=np.flatnonzero(~weak_node);weak_pos=np.flatnonzero(weak_node)
        strong_node_ids=free_node_ids[strong_pos];all_xyz=np.asarray(mesh["node_xyz"])[free_node_ids]
        strong_xyz=np.asarray(mesh["node_xyz"])[strong_node_ids]
        node_rows=[];node_cols=[];node_data=[]
        strong_col=np.full(len(free_node_ids),-1,dtype=np.int32);strong_col[strong_pos]=np.arange(len(strong_pos),dtype=np.int32)
        node_rows.extend(strong_pos.tolist());node_cols.extend(strong_col[strong_pos].tolist());node_data.extend(np.ones(len(strong_pos)).tolist())
        b_affine=np.asarray([1.0,0.0,0.0,0.0]);max_row_error=0.0;max_coord_error=0.0;max_abs_weight=0.0;max_distance=0.0
        conn_active=conn[point_solid];maximum_local_radius=3.5*float(mesh["h"])
        for row in weak_pos:
            weak_global=int(free_node_ids[row])
            sharing=conn_active[np.any(conn_active==weak_global,axis=1)]
            local_global=np.unique(sharing.reshape(-1))
            local_position=node_to_free[local_global]
            local_position=local_position[(local_position>=0)&(~weak_node[np.maximum(local_position,0)])]
            candidate_pool=np.unique(strong_col[local_position])
            if len(candidate_pool)==0:
                return {"status":"invalid_no_local_strong_support_for_weak_node","weak_support_diagonal_ratio":float(weak_support_ratio),
                    "weak_node_position":int(row),"all_original_loads_retained":True}
            pool_xyz=strong_xyz[candidate_pool];pool_distance=np.linalg.norm(pool_xyz-all_xyz[row],axis=1)
            within=pool_distance<=maximum_local_radius
            candidate_pool=candidate_pool[within];pool_distance=pool_distance[within]
            if len(candidate_pool)<4:
                return {"status":"invalid_local_aggregate_radius_has_fewer_than_four_nodes","weak_support_diagonal_ratio":float(weak_support_ratio),
                    "weak_node_position":int(row),"local_candidate_count":int(len(candidate_pool)),"maximum_local_radius_m":maximum_local_radius,
                    "all_original_loads_retained":True}
            ordered=np.argsort(pool_distance);candidate_pool=candidate_pool[ordered];pool_distance=pool_distance[ordered]
            count=min(16,len(candidate_pool));weights=None;candidate=None;A=None;distance=None
            while count<=min(128,len(candidate_pool)):
                candidate=candidate_pool[:count];distance=pool_distance[:count]
                A=np.vstack((np.ones(len(candidate)),(strong_xyz[candidate]-all_xyz[row]).T))
                if np.linalg.matrix_rank(A,tol=1e-12)==4:
                    trial=np.linalg.lstsq(A,b_affine,rcond=1e-12)[0]
                    if np.linalg.norm(A@trial-b_affine)<=1e-9:
                        weights=trial;break
                if count==min(128,len(strong_pos)):break
                count=min(count*2,128,len(strong_pos))
            if weights is None:
                return {"status":"invalid_no_local_affine_aggregate","weak_support_diagonal_ratio":float(weak_support_ratio),
                    "weak_node_position":int(row),"candidate_count":int(count),"maximum_local_radius_m":maximum_local_radius,
                    "all_original_loads_retained":True}
            node_rows.extend([int(row)]*len(candidate));node_cols.extend(candidate.astype(int).tolist());node_data.extend(weights.tolist())
            max_row_error=max(max_row_error,abs(float(np.sum(weights))-1.0))
            max_coord_error=max(max_coord_error,float(np.max(np.abs(weights@strong_xyz[candidate]-all_xyz[row]))))
            max_abs_weight=max(max_abs_weight,float(np.max(np.abs(weights))))
            max_distance=max(max_distance,float(np.max(distance)))
        Tnode=coo_matrix((node_data,(node_rows,node_cols)),shape=(len(free_node_ids),len(strong_pos))).tocsr()
        T=kron(Tnode,sparse_eye(3,format="csr"),format="csc")
        prolongation=T
        rhs_reduced=np.asarray(T.T@rhs).reshape(-1)
        applied_before=rhs.reshape((-1,3));applied_after=rhs_reduced.reshape((-1,3))
        resultant_before=applied_before.sum(axis=0);resultant_after=applied_after.sum(axis=0)
        moment_before=np.sum(np.cross(all_xyz,applied_before),axis=0);moment_after=np.sum(np.cross(strong_xyz,applied_after),axis=0)
        aggregation_info={"weak_support_diagonal_ratio":float(weak_support_ratio),"node_diagonal_median_N_per_m":diag_scale,
            "weak_nodes_aggregated":int(len(weak_pos)),"free_nodes_before_aggregation":int(len(free_node_ids)),
            "free_nodes_after_aggregation":int(len(strong_pos)),"free_dofs_before_aggregation":int(len(dofs)),
            "free_dofs_after_aggregation":int(3*len(strong_pos)),"load_nodes_aggregated":int(np.sum(weak_node[loaded_free])),
            "affine_row_sum_max_error":max_row_error,"affine_coordinate_max_error_m":max_coord_error,
            "max_aggregation_weight_abs":max_abs_weight,"max_aggregation_neighbor_distance_m":max_distance,
            "maximum_local_aggregation_radius_m":maximum_local_radius,
            "aggregation_candidate_scope":"strong nodes sharing at least one active solid material-point stencil with each weak node; limited to 3.5 grid cells",
            "projected_applied_resultant_error_N":float(np.linalg.norm(resultant_after-resultant_before)),
            "projected_applied_moment_error_Nm":float(np.linalg.norm(moment_after-moment_before)),
            "all_original_loads_retained":True,"constraint_type":"weak free-node displacements are affine-reproducing weighted combinations of nearby strong free nodes; this changes the discrete space"}
        Kff=(T.T@Kff@T).tocsc();rhs=rhs_reduced
    warnings_seen=[];cg_info=None;cg_iterations=0;cg_residual_samples=[]
    matrix_to_solve=Kff;rhs_to_solve=rhs;scale=None
    if diagonal_equilibration:
        diagonal=np.asarray(Kff.diagonal(),dtype=np.float64)
        if np.any(~np.isfinite(diagonal)) or np.any(diagonal<=0.0):
            return {"status":"invalid_nonpositive_free_stiffness_diagonal","solver":"SciPy sparse LU with exact symmetric diagonal equilibration",
                "minimum_diagonal":float(np.nanmin(diagonal)),"nonpositive_diagonal_count":int(np.sum(diagonal<=0.0))}
        scale=1.0/np.sqrt(diagonal)
        scaling=diags(scale,format="csc")
        matrix_to_solve=(scaling@Kff@scaling).tocsc()
        rhs_to_solve=scale*rhs
    try:
        if iterative:
            rhs_norm=max(float(np.linalg.norm(rhs_to_solve)),1.0)
            def record_cg_iteration(xk):
                nonlocal cg_iterations
                cg_iterations+=1
                if cg_iterations%100==0:
                    cg_residual=float(np.linalg.norm(matrix_to_solve@xk-rhs_to_solve)/rhs_norm)
                    cg_residual_samples.append(cg_residual)
                    if cg_iterations%1000==0:
                        progress={"phase":"equilibrated_sparse_cg","iterations":cg_iterations,
                            "max_iterations":cg_maxiter,"relative_residual":cg_residual,
                            "wall_seconds":time.perf_counter()-start}
                        out_path.with_name(out_path.stem+"-cg-progress.json").write_text(json.dumps(progress,indent=2)+"\n")
            solved,cg_info=cg(matrix_to_solve,rhs_to_solve,rtol=cg_tol,atol=0.0,maxiter=cg_maxiter,callback=record_cg_iteration)
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always",MatrixRankWarning)
                solved=spsolve(matrix_to_solve,rhs_to_solve,permc_spec=permc_spec)
                warnings_seen=[str(w.message) for w in caught if issubclass(w.category,MatrixRankWarning)]
        u_solve=scale*solved if scale is not None else solved
        u_free=np.asarray(prolongation@u_solve).reshape(-1) if prolongation is not None else u_solve
    except Exception as exc:
        solved=np.full_like(rhs_to_solve,np.nan);warnings_seen=[f"{type(exc).__name__}: {exc}"]
    u_solve=scale*solved if scale is not None else solved
    u_free=np.asarray(prolongation@u_solve).reshape(-1) if prolongation is not None else u_solve
    u=np.zeros(nfull,dtype=np.float64);u[dofs]=u_free
    # Recover the full nodal internal force, including clamp reactions, from
    # the same active-point stress integration used to build K.
    active_conn=conn[point_solid];active_grad=grad[point_solid];active_vol=vol[point_solid]
    gu=np.einsum("pki,pkj->pij",u.reshape((nn,3))[active_conn],active_grad)
    strain=0.5*(gu+np.swapaxes(gu,1,2))
    sigma=lam*np.trace(strain,axis1=1,axis2=2)[:,None,None]*np.eye(3)[None,:,:]+2*mu*strain
    point_force=active_vol[:,None,None]*np.einsum("pij,pkj->pki",sigma,active_grad)
    internal_node=np.zeros((nn,3),dtype=np.float64)
    np.add.at(internal_node,active_conn.reshape(-1),point_force.reshape(-1,3))
    internal=internal_node.reshape(-1)
    operator_residual=Kff_original@u_free-rhs_original
    residual=float(np.linalg.norm(operator_residual)/max(float(np.linalg.norm(rhs_original)),1.0))
    projected_residual_vector=prolongation.T@operator_residual if prolongation is not None else operator_residual
    projected_residual=float(np.linalg.norm(projected_residual_vector)/max(float(np.linalg.norm(prolongation.T@rhs_original if prolongation is not None else rhs_original)),1.0))
    weak_residual=operator_residual.reshape((-1,3))[weak_node] if prolongation is not None else np.zeros((0,3))
    weak_node_xyz=np.asarray(mesh["node_xyz"])[free_node_ids[weak_node]] if prolongation is not None else np.zeros((0,3))
    weak_residual_resultant=np.sum(weak_residual,axis=0)
    weak_residual_moment=np.sum(np.cross(weak_node_xyz,weak_residual),axis=0) if len(weak_residual) else np.zeros(3)
    finite=bool(np.isfinite(u_free).all() and np.isfinite(residual) and np.isfinite(projected_residual))
    status=("constrained_affine_aggregate_solve_converged" if finite and projected_residual<=1e-8 else "not_converged") if prolongation is not None else ("converged" if finite and residual<=1e-8 else "not_converged")
    balance=force_balance(mesh,internal,enforce=False)
    rng=np.random.default_rng(4011);v=rng.normal(size=len(dofs));kv=K@v
    energy=float(np.dot(v,kv));sym=float(sparse_norm(K-K.T)/max(sparse_norm(K),1e-30))
    point_u=np.sum(mesh["weights"][point_solid,:,None]*u.reshape((nn,3))[active_conn],axis=1)
    point_u2=np.sum(point_u*point_u,axis=1)
    solver_name=("SciPy preconditioned CG on symmetrically diagonally equilibrated exact thresholded MPM stiffness" if iterative and scale is not None else
                 "SciPy sparse LU with exact symmetric diagonal equilibration of thresholded MPM stiffness" if scale is not None else
                 "SciPy sparse LU of exact thresholded MPM material-point stiffness")
    audit={"status":status,"solver":solver_name,
        "void_stiffness":0.0,"void_inertia":0.0,"threshold":0.5,
        "active_voxels":int(solid.sum()),"active_integration_points":int(len(point_ids)),
        "unsupported_nodes_eliminated":int(np.sum(~supported)),"fixed_nodes":int(fixed.sum()),
        "free_supported_nodes":int(free.sum()),"loaded_nodes":int(np.sum(loaded)),
        "all_loaded_nodes_supported":loaded_supported,"free_dofs":int(len(dofs)),**aggregation_info,
        "sparse_matrix_nonzeros":int(K.nnz),"assembled_coo_entries":int(rows_count),
        "symmetric_diagonal_equilibration":bool(scale is not None),
        "equilibration_scale_min_max":None if scale is None else [float(scale.min()),float(scale.max())],
        "cg_info":cg_info,"cg_iterations":cg_iterations,"cg_relative_residual_samples_every_100_iterations":cg_residual_samples,
        "sparse_lu_permutation_ordering":None if iterative else permc_spec,
        "relative_linear_residual":residual,"projected_relative_linear_residual":projected_residual,
        "weak_constraint_residual_norm_N":float(np.linalg.norm(operator_residual.reshape((-1,3))[weak_node])) if prolongation is not None else 0.0,
        "weak_constraint_residual_resultant_N":weak_residual_resultant.tolist(),
        "weak_constraint_residual_moment_about_origin_Nm":weak_residual_moment.tolist(),
        "compliance_Nm":float(np.dot(rhs_original,u_free)),
        "internal_energy_quadratic_form_J":float(np.dot(u,internal)),
        "operator_symmetry_relative_error":sym,"operator_positive_random_energy":bool(energy>0),
        "matrix_rank_warnings":warnings_seen,"force_and_reaction_balance":balance,
        "threshold_topology":design_checks(solid,mesh["shape"],mesh["fixed_cell"],mesh["grid"]),
        "active_particle_displacement_max_m":float(np.sqrt(point_u2).max()),
        "active_particle_displacement_rms_m":float(np.sqrt(np.mean(point_u2))),
        "assembly_and_solve_wall_seconds":time.perf_counter()-start}
    save_array(out_path,rho_binary=solid.astype(np.float32),displacement=u.reshape((-1,3)))
    return audit


def refined_forward(grid, source_grid, q_initial, q_optimized, out):
    """Independent same-load solves on a refined background/particle grid."""
    mesh=geometry(grid)
    filter_radius = FILTER_RADIUS_CELLS*grid/source_grid
    solver=build_solver(mesh, 12.0, VF, 1200, 1e-8, filter_radius_cells=filter_radius)
    # q is stored as a flattened control vector, but the field interpolation
    # must respect the three-dimensional source voxel lattice.
    coarse_shape=(round(LENGTH*source_grid), round(HEIGHT*source_grid), round(THICKNESS*source_grid))
    q0=np.asarray(q_initial).reshape(coarse_shape)
    q1=np.asarray(q_optimized).reshape(coarse_shape)
    def resize_trilinear(source,target_shape):
        axes=[]
        for old,new in zip(source.shape,target_shape):
            x=np.clip((np.arange(new)+0.5)*old/new-0.5,0,old-1)
            lo=np.floor(x).astype(int);hi=np.minimum(lo+1,old-1);axes.append((lo,hi,x-lo))
        outfield=np.zeros(target_shape,dtype=np.float64)
        for a in (0,1):
            for b in (0,1):
                for c in (0,1):
                    ia=axes[0][0] if a==0 else axes[0][1]
                    ib=axes[1][0] if b==0 else axes[1][1]
                    ic=axes[2][0] if c==0 else axes[2][1]
                    wa=(1-axes[0][2]) if a==0 else axes[0][2]
                    wb=(1-axes[1][2]) if b==0 else axes[1][2]
                    wc=(1-axes[2][2]) if c==0 else axes[2][2]
                    outfield+=source[ia[:,None,None],ib[None,:,None],ic[None,None,:]]*wa[:,None,None]*wb[None,:,None]*wc[None,None,:]
        return outfield
    qhi0=resize_trilinear(q0,mesh["shape"]).reshape(-1)
    qhi1=resize_trilinear(q1,mesh["shape"]).reshape(-1)
    results=[]
    for label,q in (("initial",qhi0),("optimized",qhi1)):
        comp,vf,res,rho,u,vm=solver["responses"](jnp.asarray(q))
        comp.block_until_ready()
        field=np.asarray(rho).reshape(mesh["shape"])
        audit=design_checks(field,mesh["shape"],mesh["fixed_cell"],grid)
        internal=solver["action"](rho,u)
        bin_audit=solve_binary_audit(solver,mesh,rho,out/(f"refined-{label}-binary.npz"))
        zero_void_audit=None
        if label=="optimized":
            zero_void_audit=solve_true_void_binary(solver,mesh,field,out/"refined-optimized-true-void.npz")
        save_array(out/(f"refined-{label}.npz"),q=q,rho=field,displacement=np.asarray(u).reshape((-1,3)))
        results.append({"design":label,"grid":grid,"beam_cells":mesh["shape"],"particles":len(mesh["points"]),
            "compliance_Nm":float(comp),"relative_to_initial":None,"volume_fraction":float(vf),
            "mass_kg":float(vf)*RHO_BASE*LENGTH*HEIGHT*THICKNESS,
            "relative_cg_residual":float(res),"max_constituent_von_mises_pa":float(vm),
            "force_balance":force_balance(mesh,internal),
            "continuous_topology":audit,"thresholded_binary_resolve":bin_audit,
            "zero_void_binary_resolve":zero_void_audit})
    results[1]["relative_to_initial"]=results[1]["compliance_Nm"]/results[0]["compliance_Nm"]
    results[1]["response_reduction_percent"]=(1-results[1]["relative_to_initial"])*100.0
    return results


def run_optimize(args):
    mesh = geometry(args.grid)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    history_path = out/"history.csv"
    beta_schedule=(1.5, 3.0, 6.0, 12.0)
    q = jnp.full((int(np.prod(mesh["shape"])),), 0.38, dtype=jnp.float64)
    all_hist = []
    snapshots=[];snapshot_beta=[];snapshot_iteration=[]
    resume_stage=0;resume_iteration=0;resume_m=None;resume_v=None;resume_step=0
    resume_dir=Path(args.resume_dir) if args.resume_dir else None
    if resume_dir:
        checkpoint=np.load(resume_dir/"optimizer-state.npz")
        q=jnp.asarray(checkpoint["q"]);resume_m=jnp.asarray(checkpoint["m"]);resume_v=jnp.asarray(checkpoint["v"])
        resume_step=int(checkpoint["step"])
        evolution=np.load(resume_dir/"design-evolution.npz")
        snapshots=[x.copy() for x in evolution["rho_history"]]
        snapshot_beta=[float(x) for x in evolution["beta"]]
        snapshot_iteration=[int(x) for x in evolution["iteration"]]
        last_beta=snapshot_beta[-1];resume_iteration=snapshot_iteration[-1]
        resume_stage=min(range(len(beta_schedule)),key=lambda i:abs(beta_schedule[i]-last_beta))
        if "stage" in checkpoint: resume_stage=int(checkpoint["stage"])
        if "iteration" in checkpoint: resume_iteration=int(checkpoint["iteration"])
        with (resume_dir/"history.csv").open(newline="",encoding="utf-8") as f:
            prior=list(csv.DictReader(f))
        all_hist=[r for r in prior if int(r["stage"])<resume_stage or (int(r["stage"])==resume_stage and int(r["iteration"])<=resume_iteration)]
        # The checkpoint and evolution snapshot are written together every five
        # accepted/rejected updates; discard any later uncheckpointed CSV rows.
        init_npz=np.load(resume_dir/"initial-state.npz")
        (out/"initial-state.npz").write_bytes((resume_dir/"initial-state.npz").read_bytes())
        base=float(np.dot(np.asarray(mesh["load"]).reshape(-1),init_npz["u"].reshape(-1)))
        for name in ("history.csv","optimizer-state.npz","design-evolution.npz","gradient-and-solver-check.json"):
            (out/name).write_bytes((resume_dir/name).read_bytes())
    else:
        base=None
    start = time.time()
    for stage, beta in enumerate(beta_schedule):
        if stage<resume_stage: continue
        solver = build_solver(mesh, beta, VF, args.cg_maxiter, args.cg_tol)
        vg = solver["value_grad"]
        if not resume_dir:
            value, grad = vg(q)
            value.block_until_ready()
        if not resume_dir and stage == 0:
            rho = solver["rho"](q)
            opcheck = check_operator(solver, rho)
            base = float(value)
            direction = jax.random.normal(jax.random.PRNGKey(991+args.grid), q.shape, dtype=q.dtype)
            direction = direction/jnp.linalg.norm(direction)
            h = args.fd_step
            fd = (solver["objective"](q+h*direction)-solver["objective"](q-h*direction))/(2*h)
            ad = jnp.vdot(grad, direction)
            fd_err = abs(float(fd-ad))/max(abs(float(fd)), abs(float(ad)), 1e-12)
            init_response, init_vf, init_res, init_rho, init_u, init_vm = solver["responses"](q)
            init_internal=solver["action"](init_rho,init_u)
            init_json = {"objective": float(init_response), "volume_fraction": float(init_vf),
                         "relative_cg_residual": float(init_res), "max_constituent_von_mises_pa": float(init_vm),
                         "force_balance":force_balance(mesh,init_internal),
                         "operator_check": opcheck, "ad_fd_directional_relative_error": fd_err}
            (out/"gradient-and-solver-check.json").write_text(json.dumps(init_json, indent=2)+"\n")
            if not (opcheck["positive_energy"] and opcheck["random_vector_symmetry_relative_error"] < 1e-9 and fd_err < 2e-4 and float(init_res) < args.cg_tol*10):
                raise RuntimeError("operator/AD/FD/CG qualification failed")
            save_array(out/"initial-state.npz", q=np.asarray(q), rho=np.asarray(init_rho), u=np.asarray(init_u))
            snapshots.append(np.asarray(init_rho,dtype=np.float32).reshape(mesh["shape"]))
            snapshot_beta.append(beta);snapshot_iteration.append(0)
            print(json.dumps({"event":"qualified","grid":args.grid,"checks":init_json}, separators=(",",":")), flush=True)
        if resume_dir and stage==resume_stage:
            m=resume_m;v=resume_v;step=resume_step;first_iteration=resume_iteration
        else:
            m = jnp.zeros_like(q); v = jnp.zeros_like(q); step = 0;first_iteration=0
        for k in range(first_iteration,args.iterations):
            value, grad = vg(q)
            value.block_until_ready()
            m = 0.9*m+0.1*grad; v=0.999*v+0.001*grad*grad; step+=1
            mh=m/(1-0.9**step); vh=v/(1-0.999**step)
            direction=-mh/(jnp.sqrt(vh)+1e-8)
            old=float(value)
            lr=args.learning_rate
            accepted=False
            trial_q=q
            trial_value=old
            for _ in range(8):
                candidate=jnp.clip(q+lr*direction, 1e-4, 1-1e-4)
                candidate_value=solver["objective"](candidate)
                candidate_value.block_until_ready()
                trial_value=float(candidate_value)
                if np.isfinite(trial_value) and trial_value <= old*(1+1e-10):
                    trial_q=candidate; accepted=True; break
                lr*=0.5
            if accepted:
                q=trial_q
            comp,vf,res,rho,u,vm=solver["responses"](q)
            comp.block_until_ready()
            rec={"stage":stage,"beta":beta,"iteration":k+1,"accepted":accepted,"step":lr,
                 "objective":float(comp),"relative_to_initial":float(comp)/base,"volume_fraction":float(vf),
                 "relative_cg_residual":float(res),"max_constituent_von_mises_pa":float(vm),
                 "grad_inf_norm":float(jnp.max(jnp.abs(grad))),"wall_seconds":time.time()-start}
            if not np.isfinite(rec["objective"]) or rec["relative_cg_residual"]>1e-6 or abs(rec["volume_fraction"]-VF)>1e-7:
                raise RuntimeError("optimizer state failed finite/CG/volume guard")
            all_hist.append(rec)
            snapshots.append(np.asarray(rho,dtype=np.float32).reshape(mesh["shape"]))
            snapshot_beta.append(beta);snapshot_iteration.append(k+1)
            if k%1==0:
                with history_path.open("w", newline="", encoding="utf-8") as f:
                    w=csv.DictWriter(f, fieldnames=list(all_hist[0]));w.writeheader();w.writerows(all_hist)
            if (k+1)%5==0 or k+1==args.iterations:
                save_array(out/"optimizer-state.npz", q=np.asarray(q), m=np.asarray(m), v=np.asarray(v), step=np.asarray(step),stage=np.asarray(stage),iteration=np.asarray(k+1),beta=np.asarray(beta))
                save_array(out/"design-evolution.npz",rho_history=np.stack(snapshots),beta=np.asarray(snapshot_beta),iteration=np.asarray(snapshot_iteration))
                print(json.dumps({"event":"iteration",**rec},separators=(",",":")),flush=True)
    solver=build_solver(mesh, beta_schedule[-1], VF, args.cg_maxiter, args.cg_tol)
    comp,vf,res,rho,u,vm=solver["responses"](q)
    density=np.asarray(rho).reshape(mesh["shape"])
    if not np.isfinite(float(comp)) or float(res)>1e-6 or abs(float(vf)-VF)>1e-7:
        raise RuntimeError("final continuous equilibrium failed finite/CG/volume guard")
    checks=design_checks(density,mesh["shape"],mesh["fixed_cell"],args.grid)
    final_internal=solver["action"](rho,u)
    initial_npz=np.load(out/"initial-state.npz")
    initial_internal=solver["action"](jnp.asarray(initial_npz["rho"]),jnp.asarray(initial_npz["u"]).reshape(-1))
    binary_audit=solve_binary_audit(solver,mesh,rho,out/"thresholded-final.npz")
    summary={"name":"3D classical cantilever SIMP with static MPM equilibrium",
        "method":"linear-elastic small-strain energy integrated at material points using quadratic B-spline MPM shape gradients; matrix-free CG and implicit JAX linear solve gradients",
        "grid_background":args.grid,"beam_cells":mesh["shape"],"particle_quadrature_points":int(len(mesh["points"])),
        "material_voxels":int(np.prod(mesh["shape"])),"design_density_variables":int(np.sum(mesh["design_cell"])),
        "non_design_clamp_pad_voxels":int(np.sum(mesh["clamp_cell"])),"non_design_load_pad_voxels":int(np.sum(mesh["load_cell"])),
        "beam_dimensions_m":[LENGTH,HEIGHT,THICKNESS],"material":{"alloy":"aluminum benchmark assumption","E_pa":E0,"nu":NU,"density_kg_m3":RHO_BASE,"simp_penalty":3,"residual_modulus_fraction":RHO_FLOOR},
        "loads":{"clamp":"fixed x=0 grid face; first two full-section cell layers are non-design solid","free_end_pad":"fixed non-design central pad at x=L, y-width 0.125m, z-thickness 0.08333m","resultant_force_N":[0,-FORCE_N,0]},
        "load_and_reaction_audit":force_balance(mesh,final_internal),
        "initial_load_and_reaction_audit":force_balance(mesh,initial_internal),
        "volume_fraction_target":VF,"filter_radius_cell_units":FILTER_RADIUS_CELLS,
        "filter_radius_m":FILTER_RADIUS_CELLS/args.grid,"projection_beta_schedule":list(beta_schedule),
        "initial_compliance_Nm":float(np.dot(np.load(out/"initial-state.npz")["u"].reshape(-1),mesh["load"].reshape(-1))),
        "final_compliance_Nm":float(comp),"final_volume_fraction":float(vf),"final_relative_cg_residual":float(res),
        "final_peak_constituent_von_mises_pa":float(vm),"topology_audit":checks,
        "thresholded_binary_resolve":binary_audit,
        "updates_requested":args.iterations*len(beta_schedule),"updates_accepted":sum((str(row["accepted"]).lower() in ("true","1")) for row in all_hist),
        "convergence_claim":False,"history_rows":len(all_hist),
        "resumed_from_checkpoint":str(resume_dir) if resume_dir else None,
        "limitations":["SIMP density is a continuous ersatz material field; thresholded output is an audit, not a certified manufacturable part.","The linear elastic static solve is a small-strain quasi-static material-point formulation."]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    save_array(out/"final-design.npz", q=np.asarray(q), rho=density, displacement=np.asarray(u).reshape((-1,3)))
    if args.forward_grid:
        if args.forward_grid <= args.grid or args.forward_grid%8: raise ValueError("forward grid must be finer and divisible by 8")
        summary["independent_refined_forward"]=refined_forward(args.forward_grid,args.grid,np.full_like(np.asarray(q),0.38),np.asarray(q),out)
        (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"event":"complete","summary":summary},separators=(",",":")),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--grid",type=int,default=48)
    p.add_argument("--iterations",type=int,default=12)
    p.add_argument("--learning-rate",type=float,default=0.035)
    p.add_argument("--cg-maxiter",type=int,default=600)
    p.add_argument("--cg-tol",type=float,default=1e-8)
    p.add_argument("--fd-step",type=float,default=2e-4)
    p.add_argument("--forward-grid",type=int,default=0)
    p.add_argument("--resume-dir",default="")
    p.add_argument("--true-void-input",default="",help="run only a strict zero-stiffness audit from a saved density NPZ")
    p.add_argument("--true-void-maxiter",type=int,default=8000)
    p.add_argument("--true-void-tol",type=float,default=1e-8)
    p.add_argument("--true-void-resume-checkpoint",default="")
    p.add_argument("--true-void-method",choices=("jax-block-cg","scipy-sparse-direct","scipy-sparse-equilibrated-direct","scipy-sparse-equilibrated-cg"),default="jax-block-cg")
    p.add_argument("--true-void-ordering",choices=("MMD_AT_PLUS_A","COLAMD","MMD_ATA","NATURAL"),default="MMD_AT_PLUS_A")
    p.add_argument("--true-void-weak-support-ratio",type=float,default=0.0,
                   help="diagnostic only: affine-aggregate free nodes with diagonal stiffness below this ratio of the free-node median")
    p.add_argument("--output",default="outputs/classic-cantilever")
    a=p.parse_args()
    if not jax.config.jax_enable_x64: raise RuntimeError("run with JAX_ENABLE_X64=true")
    if a.grid<24 or a.grid%8: raise ValueError("grid must be >=24 and divisible by 8")
    if a.true_void_input:
        grid=a.forward_grid or a.grid
        if grid<24 or grid%8: raise ValueError("true-void grid must be >=24 and divisible by 8")
        source=np.load(a.true_void_input)
        if "rho" not in source: raise ValueError("true-void input NPZ must contain a rho density field")
        mesh=geometry(grid);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        if a.true_void_method in ("scipy-sparse-direct","scipy-sparse-equilibrated-direct","scipy-sparse-equilibrated-cg"):
            if a.true_void_weak_support_ratio<0.0:raise ValueError("weak support ratio must be nonnegative")
            audit=solve_true_void_sparse(mesh,source["rho"],out/"true-void-final.npz",
                                         diagonal_equilibration=(a.true_void_method!="scipy-sparse-direct"),
                                         iterative=(a.true_void_method=="scipy-sparse-equilibrated-cg"),
                                         cg_tol=1e-10,cg_maxiter=a.true_void_maxiter,permc_spec=a.true_void_ordering,
                                         weak_support_ratio=a.true_void_weak_support_ratio)
        else:
            audit=solve_true_void_binary(None,mesh,source["rho"],out/"true-void-final.npz",
                                         maxiter=a.true_void_maxiter,tol=a.true_void_tol,
                                         resume_checkpoint=a.true_void_resume_checkpoint or None)
        (out/"true-void-audit.json").write_text(json.dumps(audit,indent=2)+"\n")
        print(json.dumps({"event":"true_void_complete","audit":audit},separators=(",",":")),flush=True)
        return
    run_optimize(a)


if __name__=="__main__": main()
