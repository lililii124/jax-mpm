"""Compare archived and streamed frames across an actual JC restart."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from dataclasses import replace


def check(output):
    import jax
    jax.config.update('jax_enable_x64', True)
    import numpy as np
    from examples.common import run
    from examples.jc_necking import make_case

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    case = make_case(grid=50, ppc=1, frames=4, end_time=20e-9)
    substeps = round(case.duration / (4 * case.configs[0].dt))

    def solve(folder, archive, current_case=case, **kwargs):
        return run(current_case.states, current_case.materials, current_case.configs, frames=4,
                   substeps=substeps, output=folder, label='Streaming restart',
                   observer=current_case.observer, metadata=current_case.description,
                   render_output=False, write_trajectory_archive=archive, **kwargs)

    def same_arrays(first, second):
        with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    archived, streamed = output/'archived', output/'streamed'
    _, reference = solve(archived, True)
    _, prefix = solve(streamed, False, stop_after_frame=2)
    assert prefix['segment_complete'] and not prefix['completed']
    assert not list(streamed.glob('trajectory*'))
    corrupt = output/'corrupt'
    shutil.copytree(streamed, corrupt)
    missing = corrupt/'frames_data/frame_00001.npz'
    missing.unlink()
    try:
        solve(corrupt, False, resume=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Streaming restart bypassed the saved-prefix guard')
    _, result = solve(streamed, False, resume=True)
    assert reference['completed'] and result['completed']
    assert not (streamed/'trajectory.npz').exists()
    assert not list(streamed.glob('trajectory_*.npy'))
    assert json.loads((streamed/'trajectory-storage.json').read_text())['saved_frames'] == 5
    for index in range(5):
        same_arrays(archived/f'frames_data/frame_{index:05d}.npz', streamed/f'frames_data/frame_{index:05d}.npz')
    same_arrays(archived/'final_state_0.npz', streamed/'final_state_0.npz')
    for filename in ('diagnostics.csv', 'observables.csv'):
        assert (archived/filename).read_bytes() == (streamed/filename).read_bytes()

    # A second resume starts at the already-saved final frame. The one-frame
    # stream buffer is intentionally uninitialized until a new step is taken.
    _, terminal_resume = solve(streamed, False, resume=True)
    assert terminal_resume['completed'] and terminal_resume['steps'] == reference['steps']
    assert not list(streamed.glob('trajectory_*.npy'))
    same_arrays(archived/'final_state_0.npz', streamed/'final_state_0.npz')
    for index in range(5):
        same_arrays(archived/f'frames_data/frame_{index:05d}.npz', streamed/f'frames_data/frame_{index:05d}.npz')

    # Seed one failed ballistic particle so the boolean history field is
    # exercised as true through a saved prefix and its resumed continuation.
    state = case.states[0]
    failed = state.history.failed.at[0].set(True)
    failed_state = state._replace(history=state.history._replace(failed=failed))
    failed_case = replace(case, states=(failed_state,))
    failed_archived, failed_streamed = output/'ballistic-archived', output/'ballistic-streamed'
    _, failed_reference = solve(failed_archived, True, current_case=failed_case)
    _, failed_prefix = solve(failed_streamed, False, current_case=failed_case, stop_after_frame=2)
    assert failed_prefix['segment_complete'] and not failed_prefix['completed']
    _, failed_result = solve(failed_streamed, False, current_case=failed_case, resume=True)
    assert failed_reference['completed'] and failed_result['completed']
    assert not (failed_streamed/'trajectory.npz').exists()
    assert not list(failed_streamed.glob('trajectory_*.npy'))
    for index in range(5):
        first = failed_archived/f'frames_data/frame_{index:05d}.npz'
        second = failed_streamed/f'frames_data/frame_{index:05d}.npz'
        same_arrays(first, second)
        with np.load(second, allow_pickle=False) as frame:
            assert frame['failed'].dtype == np.uint8 and np.any(frame['failed'] == 1)
    same_arrays(failed_archived/'final_state_0.npz', failed_streamed/'final_state_0.npz')
    with np.load(failed_streamed/'checkpoint_0_0.npz', allow_pickle=False) as terminal:
        assert np.any(terminal['history_failed'] == 1)

    proof = dict(status='passed', particles=len(case.states[0].x), frames=5,
                 streamed_resume_matches_archived_run=True, full_material_state_equal=True,
                 prefix_validation_retained=True, derived_history_arrays_created=False,
                 terminal_resume_zero_steps_matches=True,
                 ballistic_failed_flag_true_across_split_resume=True,
                 common_sha256=hashlib.sha256((Path(__file__).parents[1]/'examples/common.py').read_bytes()).hexdigest())
    (output/'review.json').write_text(json.dumps(proof, indent=2)+'\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(check(parser.parse_args().output)))
