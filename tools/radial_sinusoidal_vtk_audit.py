"""Executable-only private-module launcher for actual saved-VTK and rigid-mesh auditing."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--review-sha256',required=True)
    parser.add_argument('--producer',type=Path,required=True)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--expected-particles',type=int,default=367744)
    args=parser.parse_args()
    review=args.review.resolve();producer=args.producer.resolve();folder=args.folder.resolve();receipt=args.receipt.resolve()
    target=folder/'vtk-full-mesh-audit.json'
    if target.exists() or receipt.exists():raise ValueError('Fresh actual VTK audit and completion receipt required')
    if digest(review)!=args.review_sha256:raise ValueError('Authenticated review source differs')
    spec=importlib.util.spec_from_file_location('radial_private_actual_vtk_audit',review)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.vtk_mesh_audit(folder,producer=producer,expected_particles=args.expected_particles)
    data=json.loads(target.read_text())
    if not data.get('passed') or data['actual_particle_frames_checked']!=51 or data['particles_per_frame']!=args.expected_particles:
        raise ValueError('Actual saved VTK audit did not complete')
    result={'schema':'radial-actual-vtk-child-v1','passed':True,'output_file':str(target),
        'output_sha256':digest(target),'review_file':str(review),'review_sha256':args.review_sha256,
        'producer':str(producer),'folder':str(folder),'expected_particles':args.expected_particles,
        'actual_particle_frames_checked':51,'renderer_overlay':data['renderer_overlay']}
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('RADIAL_ACTUAL_VTK_AUDIT_COMPLETE '+str(target),flush=True)


# pvpython can report exit 0 after Python exceptions. This executable has an
# explicit entrypoint and forces failed audits to a nonzero native exit code.
try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush();sys.stderr.flush()
    os._exit(1)
