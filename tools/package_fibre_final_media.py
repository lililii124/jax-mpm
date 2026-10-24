"""Bound and independently verify complete fibre raw archive and reviewed media."""
import argparse,hashlib,io,json,shutil,tarfile
from pathlib import Path,PurePosixPath
LIMIT=8*1024**3
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def record(path):return {'bytes':path.stat().st_size,'sha256':digest(path)}
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def safe(root,name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name:raise ValueError('Unsafe member path')
 out=root.joinpath(*p.parts)
 if not out.resolve().is_relative_to(root.resolve()) or out.is_symlink():raise ValueError('Unsafe member target')
 return out
class Parts(io.RawIOBase):
 def __init__(self,paths):self.paths=iter(paths);self.stream=None
 def readable(self):return True
 def readinto(self,b):
  while True:
   if self.stream is None:
    try:self.stream=next(self.paths).open('rb')
    except StopIteration:return 0
   n=self.stream.readinto(b)
   if n:return n
   self.stream.close();self.stream=None
 def close(self):
  if self.stream:self.stream.close()
  super().close()
def verify(root):
 manifest=read(root/'delivery-manifest.json');archive=read(root/'raw-archive/archive-manifest.json')
 if manifest['passed'] is not True or archive['all_original_raw_files_archived'] is not True:raise ValueError('Unaccepted delivery')
 if archive['compressed_bytes']>LIMIT or archive['uncompressed_file_bytes']>16*1024**3:raise ValueError('Raw archive budget exceeded')
 files=manifest['files'];expected=set(files)|{'delivery-manifest.json'}
 whole=hashlib.sha256();count=0;paths=[]
 for item in archive['parts']:
  if not 0<item['bytes']<=192*1024**2:raise ValueError('Raw part budget exceeded')
  name='raw-archive/'+item['file'];path=safe(root,name)
  if record(path)!={k:item[k] for k in ('bytes','sha256')}:raise ValueError('Changed raw part')
  expected.add(name);paths.append(path)
  with path.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):whole.update(b);count+=len(b)
 if count!=archive['compressed_bytes'] or whole.hexdigest()!=archive['sha256'] or manifest['complete_raw_archive_sha256']!=whole.hexdigest():raise ValueError('Changed whole raw archive')
 actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
 if actual!=expected or sum(safe(root,n).stat().st_size for n in actual)>LIMIT:raise ValueError('Delivery inventory/budget differs')
 for name,r in files.items():
  if record(safe(root,name))!=r:raise ValueError('Changed media/receipt '+name)
 seen=set()
 with io.BufferedReader(Parts(paths)) as stream,tarfile.open(fileobj=stream,mode='r|gz') as tar:
  for member in tar:
   safe(root,member.name)
   if not member.isfile() or member.name in seen or member.name not in archive['members']:raise ValueError('Unexpected raw member')
   seen.add(member.name);r=archive['members'][member.name];h=hashlib.sha256();n=0
   with tar.extractfile(member) as f:
    for b in iter(lambda:f.read(1024**2),b''):h.update(b);n+=len(b)
   if n!=r['bytes'] or h.hexdigest()!=r['sha256']:raise ValueError('Changed original raw member')
 if seen!=set(archive['members']):raise ValueError('Missing original raw member')
 return {'schema':'fibre-final-retrieved-whole-raw-v1','passed':True,'raw_archive_sha256':whole.hexdigest(),'original_members_checked':len(seen),'delivery_files_checked':len(actual),'original_receipt_sha256':archive['receipt_sha256'],'public_visual_review_passed':False}
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['package','verify']);p.add_argument('--output',type=Path);p.add_argument('--delivery',required=True,type=Path);a=p.parse_args();root=a.delivery.resolve()
 if a.phase=='package':
  source=a.output.resolve()
  if root.exists() or source==root or source in root.parents or root in source.parents:raise ValueError('Separate new delivery required')
  m=read(source/'delivery-manifest.json');archive=read(source/'raw-archive/archive-manifest.json');names=set(m['files'])|{'delivery-manifest.json'}|{'raw-archive/'+r['file'] for r in archive['parts']}
  if sum(safe(source,n).stat().st_size for n in names)>LIMIT:raise ValueError('Delivery exceeds8GiB')
  for n in names:
   target=safe(root,n);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(safe(source,n),target)
  verify(root)
 else:
  r=verify(root);(root/'retrieval-receipt.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
if __name__=='__main__':main()
