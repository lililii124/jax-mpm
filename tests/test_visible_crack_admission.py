"""Lightweight synthetic gate tests; these do not qualify the physical solver."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('crack_audit', Path(__file__).resolve().parents[1]/'tools/audit_phase_field_visible_crack.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CrackGateTests(unittest.TestCase):
    def fixture(self, root, break_path=False, damage=1.):
        x = np.stack(np.meshgrid(.25625+np.arange(20)*.0125, [.49375, .50625], .00625+np.arange(8)*.0125, indexing='ij'), -1).reshape(-1, 3)
        n = len(x)
        d = np.full(n, damage)
        if break_path:
            d[np.isclose(x[:, 0], .29375)] = 0
        gradient = np.zeros((n, 3))
        volume = np.full(n, .0125**3)
        fracture_density = .05*d*d
        np.savez(root/'particles.npz', reference=x, damage=d, damage_gradient=gradient)
        np.savez(root/'nodal_state.npz', damage=d, basis_nodes=np.arange(n)[:, None], basis_weight=np.ones((n, 1)), basis_gradient=np.zeros((n, 1, 3)), particle_volume=volume, particle_fracture_density=fracture_density)
        summary = dict(dx=.025, energy=[0, 0, float(volume@fracture_density)], boundary_work=1., maximum_kkt=0., minimum_nodal_damage_increment=0., energy_audit=dict(kinetic_balance_error=0., total_balance_error=0., constitutive_work_defect=0., p2g_loss=0., g2p_loss=0., damping_loss=0.), maximum_abs_step_audit=dict(kinetic_balance_error=0., total_balance_error=0.))
        (root/'summary.json').write_text(json.dumps(summary))
        (root/'history.csv').write_text('top_reaction\n1.0\n0.4\n')

    def test_connected_path_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertTrue(module.audit(root)['passed'])

    def test_subthreshold_and_disconnected_damage_fail(self):
        for options in (dict(damage=.989), dict(break_path=True)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root, **options)
                with self.assertRaises(RuntimeError):
                    module.audit(root)
                self.assertFalse(json.loads((root/'crack-checks.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
