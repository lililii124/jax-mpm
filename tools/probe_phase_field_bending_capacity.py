"""Production-mesh 100-vs50+50 exact restart/resource pilot, never a crack run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from prove_phase_field_bending_restart import measured_run


def probe(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-m', 'examples.phase_field_bending_persistent_grid',
               '--dx', '.025', '--length', '.05', '--density', '1', '--gc', '.005',
               '--speed', '.12', '--cfl', '.04', '--displacement', '.035']
    resources = {}
    for name, stop, restart in [('whole',100,None), ('prefix',50,None),
                               ('continued',100,output/'prefix/checkpoint_000050.npz')]:
        args = command+['--steps',str(stop),'--output',str(output/name)]
        if restart is not None:
            args += ['--restart',str(restart)]
        resources[name] = measured_run(args, name, output, timeout=600)
    checks = []
    for filename in ('frame_000100.npz','checkpoint_000100.npz'):
        with np.load(output/'whole'/filename) as whole, np.load(output/'continued'/filename) as continued:
            assert set(whole.files)==set(continued.files)
            for key in whole.files:
                assert np.array_equal(whole[key],continued[key]), f'Exact parity failed: {filename}:{key}'
                if whole[key].dtype.kind not in 'US':
                    assert np.isfinite(whole[key]).all()
                checks.append(filename+':'+key)
    with np.load(output/'whole/checkpoint_000100.npz') as state:
        binding = json.loads(str(state['binding']))
        assert binding['particles']==53144 and binding['dx']==.025
        assert binding['speed']==.12 and binding['displacement']==.035
        assert binding['required_steps']==9923 and int(state['step'])==100
    for name in resources:
        data = resources[name]
        assert data['samples']>0 and data['peak_sampled_uss_bytes']>0
        # The OS virtual-memory limit is the primary 6-GiB guard. These are
        # fail-closed sampled usage guards, not a claim of exact peak memory.
        assert max(data['peak_sampled_uss_bytes'],data['peak_sampled_rss_bytes']) < 6*2**30
        assert not data['timed_out'] and data['returncode']==0
    whole_events = resources['whole']['step_events']
    rates = []
    for a,b in zip(whole_events,whole_events[1:]):
        sa,sb = [int(event['line'].split()[0].split('=')[1]) for event in (a,b)]
        rates.append((b['elapsed_seconds']-a['elapsed_seconds'])/(sb-sa))
    assert len(rates)>=3
    # Conservative capacity forecast uses the slowest observed interval and
    # whole fresh-process overhead. It does not establish damaged-state cost.
    observed_step_span = whole_events[-1]['elapsed_seconds']-whole_events[0]['elapsed_seconds']
    fresh_process_overhead = max(0., resources['whole']['wall_seconds']-observed_step_span)
    forecast = max(rates)*1654+fresh_process_overhead
    reserve_ok = forecast <= 88*60  # 20% reserve within a 110-minute segment
    memory_reserve = max(data['peak_sampled_uss_bytes'] for data in resources.values()) <= .8*6*2**30
    hashes = {}
    for path in sorted(output.rglob('*.npz')):
        hashes[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(status='production_mesh_exact_parity_resource_pilot_passed',
        steps=100,split_step=50,particles=53144,dx=.025,target_displacement=.035,speed=.12,
        exact_array_checks=len(checks),resources=resources,
        whole_interval_seconds_per_step=rates,
        measured_startup_plus_teardown_seconds=fresh_process_overhead,
        conservative_1654_step_forecast_seconds=forecast,
        segment_wall_reserve_gate=reserve_ok,segment_sampled_uss_reserve_gate=memory_reserve,
        capacity_forecast_passed=reserve_ok and memory_reserve,
        checkpoint_physics_binding=binding,npz_sha256=hashes,
        damaged_state_capacity_qualified=False,medium_dispatch_admitted=False,
        crack_validated=False,completed_full_loading=False)
    (output/'capacity-pilot.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    probe(parser.parse_args().output)
