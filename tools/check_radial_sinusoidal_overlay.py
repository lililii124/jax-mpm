"""Actual51-state ParaView overlay smoke on a tiny nonphysical rendering fixture."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pvpython',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--vtk',action='store_true')
    args=parser.parse_args();output=args.output.resolve();producer=output/'producer';data=output/'fixture'
    review=module('radial_review_smoke',ROOT/'tools/radial_sinusoidal_full_review.py')
    overlay=module('radial_overlay_smoke',ROOT/'tools/radial_sinusoidal_paraview_overlay.py')
    if args.vtk:
        review.vtk_mesh_audit(data,producer=producer,expected_particles=192)
        return
    if output.exists():raise ValueError('Fresh smoke destination required')
    if args.pvpython is None or not args.pvpython.is_file():raise ValueError('Actual pvpython executable required')
    import numpy as np
    from mpm_jax.export import export_series
    data.mkdir(parents=True);(producer/'mpm_jax').mkdir(parents=True)
    shutil.copyfile(ROOT/'mpm_jax/paraview_scene.py',producer/'mpm_jax/paraview_scene.py')
    if overlay.digest(producer/'mpm_jax/paraview_scene.py')!=overlay.ORIGINAL_SCENE_SHA256:
        raise ValueError('Local fixture original scene differs from frozen de2 renderer')
    summary=copy.deepcopy(review.read(ROOT/'docs/results/adhesion-visible--adhesive-pull-off/summary.json'))
    summary['case']='nonphysical radial renderer fixture';summary['particles']=[192]
    summary['render_fixture']=True;summary['definition']['purpose']='nonphysical renderer fixture; not numerical evidence'
    summary['definition']['radial_sinusoid']={'amplitude_m':.008,'wavelength_m':.08,'axis':'y',
        'height':'A*(cos(2*pi*rho/wavelength)-1)','rho':'sqrt((x-center_x)^2+(z-center_z)^2)'}
    grid=np.meshgrid(np.linspace(.18,.82,8),np.linspace(.16,.3766176470588235,3),np.linspace(.18,.82,8),indexing='ij')
    reference=np.stack([axis.ravel() for axis in grid],axis=1)
    times=np.linspace(0,1.3,51);positions=np.repeat(reference[None,:,:],51,axis=0)
    rho=np.linalg.norm(reference[:,(0,2)]-.5,axis=1)
    for i,time in enumerate(times):positions[i,:,1]+=.035*np.sin(np.pi*time/1.3)**2*np.exp(-rho*rho/.02)
    (data/'frames_data').mkdir()
    for i,time in enumerate(times):np.savez_compressed(data/'frames_data'/f'frame_{i:05d}.npz',x=positions[i],time=time)
    review.save(data/'summary.json',summary)
    export_series(positions,{'body':np.zeros(192,dtype=np.int32),'speed':np.zeros((51,192))},times,data,spacing=.025,dim=3,displacement=True)
    command=[str(args.pvpython),'--force-offscreen-rendering',str(ROOT/'tools/radial_sinusoidal_paraview_overlay.py'),
        '--producer',str(producer),str(data),'--field','displacement','--resolution','400',
        '--color-range',*[str(x) for x in review.COLOR_RANGE],'--focus-bounds',*[str(x) for x in review.FOCUS]]
    subprocess.run(command,cwd=ROOT,check=True,timeout=300)
    delivery=output/'delivery';delivery.mkdir()
    receipt=delivery/'vtk-child-receipt.json'
    audit_command=review.vtk_audit_command(data,producer,args.pvpython.resolve(),receipt,expected_particles=192)
    with (delivery/'vtk-audit.log').open('wb') as stream:
        subprocess.run(audit_command,cwd=producer,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
    review.verify_vtk_child(data,receipt,expected_particles=192)
    review.encode_gif(data)
    review.copy_media_payload(data,delivery)
    rendered=review.read(data/'render.json')
    assert rendered['frames_rendered']==51 and rendered['resolution']==[400,400]
    assert rendered['axes'] is False and rendered['annotations'] is False
    assert rendered['renderer_overlay']==overlay.provenance()
    audit=review.read(data/'vtk-full-mesh-audit.json')
    result={'passed':True,'scope':'tiny nonphysical actual-render fixture; not f50 numerical acceptance',
        'actual_child_receipt_and_copy_chain_passed':True,'frames_rendered':51,'fixture_particles':192,'gif_resolution':[400,400],
        'gif_bytes':(data/'simulation.gif').stat().st_size,'axes':False,'text':False,
        'renderer_overlay':rendered['renderer_overlay'],'actual_mesh_checks':audit['actual_warped_rigid_mesh_checks'],
        'pvpython':str(args.pvpython),'solver_executed':False}
    review.save(output/'actual-overlay-smoke.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
