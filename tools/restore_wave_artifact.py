"""Verify a completed wave segment before restoring its checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=int,required=True)
    parser.add_argument('--artifact',type=int,required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Parent destination already exists')
    api='https://api.github.com/repos/lililii124/jax-mpm/actions'
    headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],
             'Accept':'application/vnd.github+json','User-Agent':'jax-mpm-wave-restore'}
    def get(path):
        with urllib.request.urlopen(urllib.request.Request(api+path,headers=headers),timeout=45) as stream:
            return json.load(stream)
    run=get('/runs/'+str(args.run))
    artifact=get('/artifacts/'+str(args.artifact))
    assert run['status']=='completed' and run['conclusion']=='success' and run['head_sha']==args.source
    assert artifact['workflow_run']['id']==args.run and artifact['workflow_run']['head_sha']==args.source
    assert not artifact['expired'] and artifact['digest']=='sha256:'+args.sha256
    assert 0<artifact['size_in_bytes']<512*1024**2
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*unused):return None
    opener=urllib.request.build_opener(NoRedirect)
    try:
        opener.open(urllib.request.Request(api+'/artifacts/'+str(args.artifact)+'/zip',headers=headers),timeout=45)
    except urllib.error.HTTPError as response:
        if response.code!=302:raise
        location=response.headers['Location']
    else:
        raise RuntimeError('Artifact endpoint did not provide the expected redirect')
    assert urllib.parse.urlsplit(location).scheme=='https'
    args.output.mkdir(parents=True)
    archive=args.output/'artifact.zip'
    digest=hashlib.sha256(); total=0
    # The signed storage request carries no GitHub authentication header.
    with urllib.request.urlopen(location,timeout=45) as source, archive.open('wb') as destination:
        while data:=source.read(1024**2):
            total+=len(data)
            if total>artifact['size_in_bytes']:raise ValueError('Artifact exceeded expected size')
            digest.update(data);destination.write(data)
    assert total==artifact['size_in_bytes'] and digest.hexdigest()==args.sha256
    folder=args.output/'data'
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
        assert sum(info.file_size for info in z.infolist())<2*1024**3
        for info in z.infolist():
            target=(folder/info.filename).resolve()
            if not target.is_relative_to(folder.resolve()) or (info.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe archive member')
        z.extractall(folder)
    assert hashlib.sha256((folder/'checkpoint.npz').read_bytes()).hexdigest()==args.checkpoint_sha256
    review=json.loads((folder/'review.json').read_text())
    assert review['commit']==args.source
    receipt=dict(run=args.run,artifact=args.artifact,source=args.source,archive_sha256=args.sha256,
                 archive_bytes=total,checkpoint_sha256=args.checkpoint_sha256,start_step=review['steps'])
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
