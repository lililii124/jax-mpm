from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, str(Path('.').resolve()))
from examples.classic_cantilever_topopt import E0, NU, geometry, solve_true_void_sparse, FORCE_N, design_checks

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--grid', type=int, choices=(48, 64), default=64)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=True)
source = args.input.read_bytes()
(out / 'source-design.npz').write_bytes(source)
with np.load(io.BytesIO(source)) as data:
    rho = data['rho'].copy()
grid = args.grid
mesh = geometry(grid)
mesh['load'] *= 50000.0 / FORCE_N
shape = mesh['shape']
checks = design_checks(rho, shape, mesh['fixed_cell'], grid)
if not checks['clamp_to_load_pad_connected'] or checks['six_connected_component_count'] != 1:
    raise RuntimeError(f'Thresholded geometry is disconnected: {checks}')
audit = solve_true_void_sparse(mesh, rho, out / 'true-void.npz', diagonal_equilibration=True,
                               iterative=True, cg_tol=1e-9, cg_maxiter=20000, weak_support_ratio=0.0)
if (out / 'true-void.npz').exists():
    with np.load(out / 'true-void.npz') as result:
        displacement = result['displacement']
        solid = result['rho_binary'].ravel() > .5
    point_solid = solid[mesh['cell_id_per_point']]
    conn = mesh['indices'][point_solid]
    grads = mesh['gradients'][point_solid]
    gu = np.einsum('pki,pkj->pij', displacement[conn], grads)
    strain = .5 * (gu + np.swapaxes(gu, 1, 2))
    lam = E0 * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E0 / (2 * (1 + NU))
    sigma = lam * np.trace(strain, axis1=1, axis2=2)[:, None, None] * np.eye(3) + 2 * mu * strain
    dev = sigma - np.trace(sigma, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    vm = np.sqrt(1.5 * np.sum(dev * dev, axis=(1, 2)))
    audit['max_active_von_mises_Pa'] = float(np.max(vm))
    audit['loaded_pad_mean_deflection_m'] = audit['compliance_Nm'] / 50000.0
    audit['stress_limit_100MPa_pass'] = bool(audit['status'] == 'converged' and audit['max_active_von_mises_Pa'] <= 1e8)
    audit['deflection_limit_2mm_pass'] = bool(audit['status'] == 'converged' and audit['loaded_pad_mean_deflection_m'] <= .002)
summary = {'source_sha256':hashlib.sha256(source).hexdigest(), 'topology':checks,
           'support': 'fixed clamp and load pads', 'audit':audit}
(out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, separators=(',', ':')), flush=True)
