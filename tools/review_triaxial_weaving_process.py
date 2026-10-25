"""Read actual raw process states; a prefix is not a completed weave."""
import argparse
import json
from pathlib import Path
import numpy as np


def crossing_measurements(state,geometry):
    """Measure current material near planned crossing sites, not guide targets."""
    centres={}
    for fibre in geometry['fibres']:
        selected=state['x'][state['fibre_id']==fibre['id']]
        centres[fibre['id']]=selected.reshape(fibre['longitudinal_sections'],-1,3).mean(axis=1)
    rows=[];radius=geometry['diameter_m']/2
    for crossing in geometry['crossings']:
        site=np.asarray(crossing['plane_xz']);measure=[]
        for fibre in crossing['fibre_ids']:
            c=centres[fibre];a=c[:-1][:,[0,2]];b=c[1:][:,[0,2]];t=b-a
            fraction=np.clip(np.sum((site-a)*t,axis=1)/np.maximum(np.sum(t*t,axis=1),1e-30),0,1)
            distance=np.linalg.norm(a+fraction[:,None]*t-site,axis=1);i=int(distance.argmin())
            height=float(c[i,1]*(1-fraction[i])+c[i+1,1]*fraction[i])
            measure.append(dict(xz_distance_m=float(distance[i]),height_y_m=height,
                finite_cord_covers_site=bool(distance[i]<=radius),section_index=i))
        covered=all(m['finite_cord_covers_site'] for m in measure)
        delta=measure[0]['height_y_m']-measure[1]['height_y_m']
        rows.append(dict(fibre_ids=crossing['fibre_ids'],site_xz_m=crossing['plane_xz'],
            actual_material_measurements=measure,both_finite_cords_present=covered,
            signed_height_difference_m=delta,expected_order=crossing['expected_left_above_right'],
            order_matches_if_present=bool(covered and delta*crossing['expected_left_above_right']>0),
            scope='Measured local order only; opening or mere planar coverage is not completed interlacing'))
    return rows


def review(folder):
    folder=Path(folder);plan=json.loads((folder/'plan.json').read_text())
    reference=np.load(folder/'reference.npz');initial=reference['reference'];masks=reference['grip_masks']
    geometry=json.loads((folder/'geometry.json').read_text())
    free=~np.any(masks,axis=0);frames=[];last_step=0
    for path in sorted(folder.glob('frame_*.npz')):
        step=int(path.stem.split('_')[-1]);state=np.load(path)
        assert np.array_equal(state['mass'],reference['mass'])
        assert np.array_equal(state['fibre_id'],reference['fibre_id'])
        assert np.array_equal(state['family_id'],reference['family_id'])
        assert all(np.isfinite(state[k]).all() for k in ['x','v','F','C','damage','mass'])
        assert np.linalg.det(state['F']).min()>0
        moved=np.linalg.norm(state['x']-initial,axis=1)
        frames.append(dict(step=step,time=float(state['time']),
            freely_updated_body_particles_moved=int(np.sum(moved[free]>1e-12)),
            maximum_free_displacement=float(moved[free].max()),
            maximum_free_F_change=float(abs(state['F'][free]-np.eye(3)).max()),
            minimum_detF=float(np.linalg.det(state['F']).min()),maximum_damage=float(state['damage'].max())))
        last_step=step
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as checkpoint:
        assert int(checkpoint['step'])==last_step
        final=np.load(folder/f'frame_{last_step:07d}.npz')
        assert np.array_equal(checkpoint['state_x'],final['x'])
    observations=json.loads((folder/'observations.json').read_text());assert observations[-1]['step']==last_step
    assert all(row['contact_valid'] and row['mpm_valid'] for row in observations)
    report=dict(status='finite_MPM_prefix_audited_not_media_qualification',steps=last_step,
        full_protocol_completed=last_step>=plan['full_steps'],matrix_present=False,
        free_particle_updates_observed=bool(frames[-1]['freely_updated_body_particles_moved']),
        actual_crossing_order_requires_full_stage_review=True,frames=frames,
        final_material_crossing_measurements=crossing_measurements(final,geometry),
        limitations=['Circular-equivalent local Hertz; extended/line/end-cap contact rejected',
            'No friction, no matrix, no industrial material/rate calibration',
            'Initial straight cords and endpoint commands do not constitute a finished weave'])
    (folder/'independent-review.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('folder');args=parser.parse_args()
    print(json.dumps(review(args.folder),indent=2))
