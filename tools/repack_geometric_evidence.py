"""Repackage authenticated numerical evidence without a numerical update.

Original tarparts are unchanged. ZIP non-payload bytes are preserved so the
original official ZIP can be reconstructed byte for byte and SHA checked locally.
"""
import argparse,base64,hashlib,json,os,struct,subprocess,urllib.request,urllib.error,urllib.parse,zipfile
from pathlib import Path
REPO='lililii124/jax-mpm'
EXPECTED_COMMIT='492591dfca855a6a526051bc7026aa4428e1f051'
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=int,required=True);p.add_argument('--artifact',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('GITHUB_REPOSITORY')!=REPO or os.environ.get('GITHUB_REF')!='refs/heads/work/metal-geometric-evidence-transport':raise RuntimeError('Exact transport-only ref required')
    root=Path(__file__).resolve().parents[1];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if head!=os.environ['GITHUB_SHA']:raise RuntimeError('Transport source differs')
    own='.github/workflows/metal-geometric-evidence-transport.yml';alias='.github/workflows/engineering-previews.yml'
    if (root/own).read_bytes()!=(root/alias).read_bytes():raise RuntimeError('Alias differs')
    for name in ('tools/repack_geometric_evidence.py',own,alias):
        if (root/name).read_bytes()!=subprocess.check_output(['git','show',head+':'+name],cwd=root):raise RuntimeError('Committed transport source differs')
    a.output.mkdir(parents=True,exist_ok=False)
    headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'}
    api=f'https://api.github.com/repos/{REPO}/actions/artifacts/{a.artifact}'
    with urllib.request.urlopen(urllib.request.Request(api,headers=headers),timeout=30) as response:meta=json.load(response)
    if meta['workflow_run']['id']!=a.run or meta['workflow_run']['head_sha']!=EXPECTED_COMMIT or meta['expired']:raise RuntimeError('Wrong frozen numerical artifact')
    if not meta['name'].startswith('metal-v6-geometric125-cfl') or not meta['digest'].startswith('sha256:') or not 0<meta['size_in_bytes']<2*1024**3:raise RuntimeError('Wrong numerical artifact scope')
    try:
        urllib.request.build_opener(NoRedirect).open(urllib.request.Request(api+'/zip',headers=headers),timeout=30);raise RuntimeError('Expected302')
    except urllib.error.HTTPError as response:
        if response.code!=302:raise
        location=response.headers['Location'];response.close()
    del headers
    parsed=urllib.parse.urlsplit(location)
    if parsed.scheme!='https' or parsed.username or parsed.password:raise RuntimeError('Invalid signed URL')
    archive=a.output.parent/'original-evidence.zip'
    with urllib.request.urlopen(location,timeout=60) as response,archive.open('wb') as out:
        total=0
        for block in iter(lambda:response.read(1024**2),b''):
            total+=len(block)
            if total>meta['size_in_bytes']:raise RuntimeError('Unexpected archive length')
            out.write(block)
    expected=meta['digest'].split(':',1)[1]
    if archive.stat().st_size!=meta['size_in_bytes'] or sha(archive)!=expected:raise RuntimeError('Whole official archive SHA failed')
    with zipfile.ZipFile(archive) as z:
        infos=sorted(z.infolist(),key=lambda info:info.header_offset)
        for info in infos:
            if Path(info.filename).name!=info.filename or info.compress_type!=zipfile.ZIP_STORED:raise RuntimeError('Expected flat uncompressed transport ZIP')
            z.extract(info,a.output)
    manifest=json.loads((a.output/'transport.json').read_text())
    for name,info in manifest['parts'].items():
        if (a.output/name).stat().st_size!=info['bytes'] or sha(a.output/name)!=info['sha256']:raise RuntimeError('Original tarpart differs')
    segments=[];cursor=0
    with archive.open('rb') as f:
        for info in infos:
            f.seek(info.header_offset);header=f.read(30);values=struct.unpack('<4s5H3I2H',header)
            if values[0]!=b'PK\x03\x04':raise RuntimeError('Bad local ZIP header')
            start=info.header_offset+30+values[-2]+values[-1]
            f.seek(cursor);literal=f.read(start-cursor)
            segments.append(dict(kind='literal',offset=cursor,bytes=len(literal),base64=base64.b64encode(literal).decode()))
            segments.append(dict(kind='file',offset=start,name=info.filename,bytes=info.file_size,sha256=sha(a.output/info.filename)))
            if info.compress_size!=info.file_size:raise RuntimeError('Unexpected payload transformation')
            cursor=start+info.file_size
        f.seek(cursor);literal=f.read();segments.append(dict(kind='literal',offset=cursor,bytes=len(literal),base64=base64.b64encode(literal).decode()))
    (a.output/'zip-structure.json').write_text(json.dumps(dict(original_bytes=meta['size_in_bytes'],original_sha256=expected,segments=segments),indent=2)+'\n')
    (a.output/'source-archive-verification.json').write_text(json.dumps(dict(source_run=a.run,source_artifact=a.artifact,source_commit=EXPECTED_COMMIT,official_archive_bytes=meta['size_in_bytes'],official_archive_sha256=expected,whole_archive_verified=True,original_transport_sha256=sha(a.output/'transport.json'),transport_only_commit=head,numerical_updates=0,unchanged_original_parts=manifest['parts']),indent=2)+'\n')
    print(json.dumps(dict(whole_original_zip_verified=True,original_bytes=meta['size_in_bytes'],parts=len(manifest['parts']),numerical_updates=0)))
if __name__=='__main__':main()
