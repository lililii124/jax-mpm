"""Authenticated display-only namespace overlay for the immutable radial ParaView scene."""
import argparse
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ORIGINAL_SCENE_SHA256='47015e792433ac1c5e86f0b271f6ce1e16ef7c0e16f9605ef57e8597c2da7539'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scene(producer):
    path=Path(producer)/'mpm_jax/paraview_scene.py'
    if digest(path)!=ORIGINAL_SCENE_SHA256:raise ValueError('Original immutable radial scene bytes differ')
    spec=importlib.util.spec_from_file_location('authenticated_radial_scene_overlay',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # pvpython may preload vtk/numpy reduction aliases into its main namespace.
    # Keep the original file byte-exact and explicitly restore Python scalar bounds.
    module.min=builtins.min
    module.max=builtins.max
    return module


def provenance():
    return {'schema':'radial-paraview-builtins-overlay-v1',
        'original_scene_sha256':ORIGINAL_SCENE_SHA256,
        'overlay_sha256':digest(Path(__file__)),
        'patch':'private module namespace; min=builtins.min; max=builtins.max',
        'numerical_source_modified':False,'geometry_force_law_modified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--producer',type=Path,required=True)
    args,forward=parser.parse_known_args()
    if not forward:raise ValueError('Original scene dataset arguments required')
    scene=load_scene(args.producer)
    sys.argv=[str(args.producer/'mpm_jax/paraview_scene.py'),*forward]
    scene.main()
    record=Path(forward[0])/'render.json'
    data=json.loads(record.read_text());data['renderer_overlay']=provenance()
    record.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
