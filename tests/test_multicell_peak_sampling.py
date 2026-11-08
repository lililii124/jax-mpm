import json

from tools.compare_multicell_binary_blast import compare


def test_comparison_uses_peaks_between_saved_frames(tmp_path):
    def replay(name, mask, endpoint, interval):
        path = tmp_path/name
        path.mkdir()
        meta = dict(settings={'steps': 2, 'dt': .1}, steps=2, frame_interval=2,
                    source_sha256={'solver': 'same'}, jax_version='same', numpy_version='same',
                    binary_mass_kg=1., geometry_sha256=mask, solid_mask_sha256=mask,
                    peak_sampling='every-step', diagnostic_only=False, connected_faces=True, detached_voxels=0)
        row = dict(valid=True, min_J=1., kinetic_J=0., elastic_J=0., pressure_work_J=0.,
                   boundary_work_J=0., energy_residual_J=0., momentum_residual=0.)
        history = [dict(row, step=0, time_s=0., back_displacement_max_m=0., back_displacement_interval_peak_m=0.),
                   dict(row, step=2, time_s=.2, back_displacement_max_m=endpoint,
                        back_displacement_interval_peak_m=interval)]
        (path/'inputs.json').write_text(json.dumps(meta))
        (path/'history.json').write_text(json.dumps(history))
        return path
    baseline = replay('baseline', 'one', 1., 3.)
    candidate = replay('candidate', 'two', 1.5, 2.)
    result = compare(baseline, candidate)
    assert result['baseline_sampled_peak_m'] == 3.
    assert result['candidate_sampled_peak_m'] == 2.
    assert result['peak_sampling'] == 'every-step'
