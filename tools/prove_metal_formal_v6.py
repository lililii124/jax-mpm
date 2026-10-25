"""Bounded source-frozen local multifield proof; no remote dispatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
FILES=('mpm_jax/fragment_multifield_v6_mpm.py','examples/metal_fragment_formal_v6.py','tools/check_multifield_accelerated_contact.py','examples/metal_fragment_multifield_v3.py','examples/metal_fragment_multifield_compact.py','examples/metal_fragment_multifield_v2.py','tools/prove_metal_formal_v6.py','examples/metal_fragment_multifield.py','tools/audit_metal_fragments.py','mpm_jax/materials/linear_metal.py','mpm_jax/materials/shock_metal.py','mpm_jax/fibre_mpm.py','mpm_jax/contact.py','mpm_jax/core.py','examples/metal_sphere_impact_fracture.py','examples/metal_sphere_geometry.py','examples/metal_plate_geometry.py','examples/woven_geometry.py')


def main():
 import psutil
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--coarse',action='store_true');p.add_argument('--resolution',choices=('formal',),default='formal');p.add_argument('--frames',type=int,default=8);p.add_argument('--epoch-ns',type=float,choices=(125.,62.5),default=125.);p.add_argument('--iterations',type=int,choices=(256,512,1024,2048),default=1024);p.add_argument('--end-ns',type=float,default=2000.);p.add_argument('--contact-tolerance',type=float,choices=(.005,.0005),default=.005);p.add_argument('--contact-proof',type=Path,default=Path('.local/multifield-v6-accelerated-contact-proof/proof.json'));args=p.parse_args()
 contact=json.loads(args.contact_proof.read_text())
 if hashlib.sha256(args.contact_proof.read_bytes()).hexdigest()!='8a38e4d0d8ebc26a1daa8366150a7c81c98dd7a389c82776b885d2ac80e94c52' or contact['solver_sha256']!=hashlib.sha256((ROOT/'mpm_jax/fragment_multifield_v6_mpm.py').read_bytes()).hexdigest():raise ValueError('Exact verified constrained contact proof/source required')
 if args.output.exists():raise FileExistsError('Fresh proof only')
 args.output.mkdir(parents=True)
 (args.output/'admitted-contact-proof.json').write_bytes(args.contact_proof.read_bytes())
 hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
 (args.output/'source.json').write_text(json.dumps(hashes,indent=2)+'\n')
 for name in FILES:
  target=args.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
 (args.output/'runtime-freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
 cmd=[sys.executable,'-m','examples.metal_fragment_formal_v6','--contact-tolerance',str(args.contact_tolerance),'--epoch-ns',str(args.epoch_ns),'--iterations',str(args.iterations),'--end-ns',str(args.end_ns),'--resolution',args.resolution,'--frames',str(args.frames),'--output',str(args.output/'result')]+(['--coarse'] if args.coarse else [])
 begun=time.monotonic();samples=[];reason=None;code=None;proc=None
 def save(name,value):
  path=args.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
 def terminate():
  if proc is None or proc.poll() is not None:return
  try:
   parent=psutil.Process(proc.pid)
   for child in parent.children(recursive=True):
    try:child.kill()
    except psutil.NoSuchProcess:pass
  except psutil.NoSuchProcess:pass
  try:proc.kill()
  except ProcessLookupError:pass
 try:
  with (args.output/'solve.log').open('wb') as log:
   env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',JAX_ENABLE_X64='true')
   proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
   while proc.poll() is None:
    try:
     parent=psutil.Process(proc.pid);usage=sum(p.memory_full_info().uss for p in [parent]+parent.children(recursive=True))
    except psutil.NoSuchProcess:
     try:
      proc.wait(timeout=.05)
      break
     except subprocess.TimeoutExpired:
      reason='Live memory accounting vanished';usage=0
    except psutil.AccessDenied:reason='USS access denied';usage=0
    raw=0
    for path in args.output.rglob('*'):
     try:
      if path.is_file():raw+=path.stat().st_size
     except FileNotFoundError:pass
    elapsed=time.monotonic()-begun;samples.append(dict(elapsed_s=elapsed,tree_uss_bytes=usage,raw_bytes=raw))
    if usage>6*1024**3:reason='6GiB USS ceiling'
    if raw>2*1024**3-256*1024**2:reason='Raw disk watermark'
    if elapsed>1800:reason='30min proof ceiling'
    save('monitor-live.json',dict(command=cmd,reason=reason,samples=samples))
    if reason:terminate();break
    time.sleep(1)
   code=proc.wait()
 except BaseException as error:
  reason=reason or ('Monitor exception: '+repr(error))
  raise
 finally:
  terminate()
  if proc is not None:code=proc.wait()
  changed=[name for name,sha in hashes.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha]
  save('monitor.json',dict(command=cmd,returncode=code,reason=reason,source_changed=changed,samples=samples))
 if code or reason or changed:raise RuntimeError('Proof rejected; complete failure evidence retained')


if __name__=='__main__':main()
