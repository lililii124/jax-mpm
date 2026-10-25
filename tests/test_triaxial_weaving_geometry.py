import numpy as np
from examples.triaxial_weaving_geometry import make_process,fixture_position


def test_unwoven_unit_has_three_cyclic_crossings_and_conserves_reference_volume():
    geometry=make_process(per_family=1)
    assert len(geometry['metadata']['crossings'])==3
    expected={(tuple(c['fibre_ids'])):c['expected_left_above_right'] for c in geometry['metadata']['crossings']}
    assert expected[(0,1)]==1 and expected[(1,2)]==1 and expected[(0,2)]==-1
    masks=np.asarray(geometry['masks'])
    assert np.max(masks.sum(0))==1 and (~masks.any(0)).sum()>6000
    analytic=sum(np.pi*f['radius_m']**2*f['reference_arclength_m'] for f in geometry['metadata']['fibres'])
    assert np.isclose(geometry['volume'].sum(),analytic,rtol=1e-14)
    assert np.array_equal(geometry['mass'],1500*geometry['volume'])
    assert all(f['feed_reserve_m']>0 for f in geometry['metadata']['fibres'])


def test_machine_knots_move_only_end_fixture_targets():
    geometry=make_process(per_family=2);meta=geometry['metadata']
    times=np.asarray(meta['fixture_times']);positions=np.asarray(meta['fixture_positions'])
    assert positions.shape==(17,12,3)
    assert np.allclose(fixture_position(times[0],times,positions),positions[0],rtol=0,atol=0)
    assert np.allclose(fixture_position(times[-1],times,positions),positions[-1],rtol=0,atol=0)
    assert not meta['matrix_present']
    assert len(geometry['x'])==13032
    for fibre in meta['fibres']:
        p=geometry['x'][geometry['fibre_id']==fibre['id']]
        # Each initial circular section lies on a straight uncrimped centreline.
        centres=p.reshape(fibre['longitudinal_sections'],meta['cross_section_particles'],3).mean(1)
        assert np.ptp(centres[:,1])<1e-15


def test_v2_feed_speed_is_bounded_by_actual_fixture_motion():
    from examples.triaxial_weaving_geometry import fixture_position
    geometry=make_process(per_family=2)
    metadata=geometry['metadata']
    for stage in metadata['stages']:
        if stage['name'].startswith('feed_yarn_'):
            middle=(stage['start']+stage['end'])/2;delta=1e-9
            velocity=(fixture_position(middle+delta,metadata['fixture_times'],metadata['fixture_positions'])-fixture_position(middle-delta,metadata['fixture_times'],metadata['fixture_positions']))/(2*delta)
            assert np.linalg.norm(velocity,axis=1).max()<=3.000001
            assert stage['end']-stage['start']>=.0036-1e-14
