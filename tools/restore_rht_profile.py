"""Restore selected original RHT state/source files after whole-parent checks."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.rht_impact_prefix import PartReader,verify,digest


def main():
    specification=json.loads((ROOT/'tools/rht_profile_parent.json').read_text())
    transfer=ROOT/'.rht-transfer';output=ROOT/'.rht-parent'
    if transfer.exists() or output.exists():raise FileExistsError('Parent destination exists')
    if shutil.disk_usage(ROOT).free<4*1024**3:raise RuntimeError('Parent restore needs 4 GiB disk headroom')
    transfer.mkdir()
    api='https://api.github.com/repos/lililii124/jax-mpm/actions'
    headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'}
    def get(path):
        with urllib.request.urlopen(urllib.request.Request(api+path,headers=headers),timeout=60) as stream:
            return json.load(stream)
    run=get('/runs/'+str(specification['run']))
    assert run['status']=='completed' and run['conclusion']=='failure' and run['head_sha']==specification['source']
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args):return None
    opener=urllib.request.build_opener(NoRedirect)
    for expected in specification['artifacts']:
        metadata=get('/artifacts/'+str(expected['id']))
        assert not metadata['expired'] and metadata['workflow_run']['id']==specification['run']
        assert metadata['digest']=='sha256:'+expected['sha256'] and metadata['size_in_bytes']==expected['bytes']
        try:opener.open(urllib.request.Request(api+'/artifacts/'+str(expected['id'])+'/zip',headers=headers),timeout=60)
        except urllib.error.HTTPError as response:
            if response.code!=302:raise
            location=response.headers['Location'];response.close()
        else:raise RuntimeError('Expected artifact redirect')
        archive=transfer/(str(expected['id'])+'.zip')
        total=0;checksum=hashlib.sha256()
        with urllib.request.urlopen(location,timeout=60) as source,archive.open('wb') as target:
            while data:=source.read(1024**2):
                total+=len(data)
                if total>expected['bytes']:raise ValueError('Artifact too large')
                checksum.update(data);target.write(data)
        assert total==expected['bytes'] and checksum.hexdigest()==expected['sha256']
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for item in z.infolist():
                destination=(transfer/item.filename).resolve()
                assert destination.is_relative_to(transfer.resolve()) and not destination.exists()
                assert '/' not in item.filename and '\\' not in item.filename
            z.extractall(transfer)
    assert digest(transfer/'transport.json')==specification['transport_sha256']
    verify(transfer)
    index=json.loads((transfer/'transport.json').read_text())
    wanted={'raw/checkpoint.json','raw/status.json','receipt.json','runtime-freeze.txt','git-blobs.json','source-manifest.json'}
    selected={}
    reader=PartReader(transfer/name for name in index['parts'])
    with tarfile.open(fileobj=reader,mode='r|') as archive:
        for member in archive:
            if member.name not in wanted and not member.name.startswith('source/'):continue
            destination=(output/member.name).resolve()
            assert destination.is_relative_to(output.resolve()) and member.isfile()
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('wb') as target:shutil.copyfileobj(archive.extractfile(member),target,1024**2)
            assert digest(destination)==index['files'][member.name]['sha256']
            selected[member.name]=index['files'][member.name]
            if member.name=='raw/checkpoint.json':
                checkpoint=json.loads(destination.read_text());assert checkpoint['frame']==3
                wanted.update('raw/'+name for name in checkpoint['states'])
    assert wanted<=selected.keys()
    old=json.loads((output/'receipt.json').read_text())
    assert old['commit']==specification['source'] and old['error']=='RuntimeError: 50 minute solver guard exceeded'
    report={'transport_sha256':specification['transport_sha256'],'all_91_parent_files_reverified':True,
            'frame':3,'physical_steps_executed':0,'selected_members':selected,'official_artifacts':specification['artifacts']}
    (output/'extraction-review.json').write_text(json.dumps(report,indent=2)+'\n')
    receipt=ROOT/'outputs/rht-profile/parent-verification.json';receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'verified_parent_files':len(index['files']),'selected_files':len(selected),'frame':3}))


if __name__=='__main__':main()
