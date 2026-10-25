"""Fresh batch-process plan; alternative time steps are local diagnostics."""
import numpy as np
from .triaxial_weaving_process import plan_process,json_value
from .triaxial_weaving_batch_geometry import make_batch_process


def plan_batch(per_family=8,nominal_cfl=.08):
    if nominal_cfl not in (.08,.24):raise ValueError('Only the two declared short diagnostic time steps are admitted')
    _,plan=plan_process(per_family=per_family)
    geometry=make_batch_process(per_family=per_family)
    plan.update(case=geometry['metadata']['case'],nominal_elastic_cfl=nominal_cfl,
        dt=nominal_cfl*plan['dx']/plan['elastic_wave_speed_m_s'],
        duration_s=geometry['metadata']['duration_s'],protocol=json_value(geometry['metadata']['stages']),
        batch_compatibility=json_value(geometry['metadata']['batch_compatibility']),
        maximum_machine_speed_m_s=3.,short_diagnostic_cfl_ceiling=.25,
        qualification='Fresh dynamic batch process; CFL.24 is unqualified until equal-time actual diagnostic, no full-process convergence claim')
    plan['full_steps']=int(np.ceil(plan['duration_s']/plan['dt']))
    assert all(s['peak_fixture_speed_m_s']<=3.+1e-12 for s in plan['protocol'])
    return geometry,plan
