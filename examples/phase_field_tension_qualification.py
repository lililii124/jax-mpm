"""Read-only case-specific small-strain checks; these are not material laws.

Only immutable, initially unit-damage quadrature points may represent a
regularized pre-existing displacement jump. Newly damaged points are never
added to that exemption. All physical states and global strains remain visible.
"""
import numpy as np

POLICY = 'initial-unit-seed-jump-v1'


def immutable_unit_seed(reference, initial_damage, dx):
    return ((initial_damage >= 1-1e-12) & (reference[:,0] <= .3+1e-12)
            & (abs(reference[:,1]-.3) <= dx+1e-12))


def inspect(gradient, displacement, damage, volume, seed):
    G=np.asarray(gradient);u=np.asarray(displacement);d=np.asarray(damage)
    volume=np.asarray(volume);seed=np.asarray(seed)
    eps=.5*(G+G.swapaxes(-1,-2));F=np.eye(3)[None]+G
    green=.5*(F.swapaxes(-1,-2)@F-np.eye(3)[None])
    left,s,right=np.linalg.svd(F)
    R=left@right
    angles=np.arccos(np.clip((np.trace(R,axis1=-2,axis2=-1)-1)/2,-1,1))
    lam=1000*.2/(1.2*.6);mu=1000/(2*1.2)
    def density(strain):
        eig=np.linalg.eigvalsh(strain);tr=eig.sum(-1)
        plus=.5*lam*np.maximum(tr,0)**2+mu*(np.maximum(eig,0)**2).sum(-1)
        minus=.5*lam*np.minimum(tr,0)**2+mu*(np.minimum(eig,0)**2).sum(-1)
        return ((1-1e-8)*(1-d)**2+1e-8)*plus+minus
    linear_energy=density(eps);green_energy=density(green)
    total=float(np.sum(volume*linear_energy))
    residual=float(np.sum(volume[seed]*linear_energy[seed]))
    nonseed=float(np.sum(volume[~seed]*linear_energy[~seed]))
    result=dict(policy=POLICY,initial_unit_seed_points=int(seed.sum()),
        global_maximum_strain=float(abs(eps).max()),
        old_global_guard_would_fail=bool(abs(eps).max()>.02),
        nonseed_maximum_strain=float(abs(eps[~seed]).max()),
        global_rotation_degrees=float(np.rad2deg(angles).max()),
        minimum_det_F=float(np.linalg.det(F).min()),
        maximum_particle_displacement_over_H=float(np.linalg.norm(u,axis=1).max()/.6),
        seed_residual_elastic_fraction=residual/max(total,1e-30),
        nonseed_absolute_geometric_energy_difference_fraction=float(
            np.sum(volume[~seed]*abs(green_energy[~seed]-linear_energy[~seed])))/max(nonseed,1e-30),
        minimum_current_seed_damage=float(d[seed].min()))
    numerical=[v for v in result.values() if isinstance(v,(float,int))]
    if not np.isfinite(numerical).all():
        raise RuntimeError('Nonfinite strain/geometry qualification')
    passed=(result['nonseed_maximum_strain']<=.02
        and result['global_rotation_degrees']<=1.
        and result['minimum_det_F']>.5
        and result['maximum_particle_displacement_over_H']<=.01
        and result['seed_residual_elastic_fraction']<=.001
        and result['nonseed_absolute_geometric_energy_difference_fraction']<=.02
        and result['minimum_current_seed_damage']>=1-1e-10)
    result['passed']=bool(passed)
    return result
