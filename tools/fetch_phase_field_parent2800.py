"""Fetch one frozen Actions artifact without forwarding GitHub credentials."""
import hashlib
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

API = 'https://api.github.com/repos/lililii124/jax-mpm/actions/artifacts/12586985399/zip'
DESTINATION = Path('outputs/phase-field-parent2400.zip')
EXPECTED = '6abf7642b447cf9b9ab4c79f33fc07eac7f1dca8df01bea513fc3807bd0986bf'
EXPECTED_BYTES = 75709261


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    DESTINATION.parent.mkdir(parents=True,exist_ok=True)
    partial=DESTINATION.with_suffix('.partial')
    deadline=time.monotonic()+180
    for attempt in range(3):
        try:
            request=urllib.request.Request(API,headers={
                'Authorization':'Bearer '+os.environ['GH_TOKEN'],
                'Accept':'application/vnd.github+json',
                'X-GitHub-Api-Version':'2022-11-28'})
            try:
                with urllib.request.build_opener(NoRedirect).open(request,timeout=min(30,deadline-time.monotonic())):
                    raise RuntimeError('Expected artifact API redirect')
            except urllib.error.HTTPError as response:
                if response.code != 302:
                    response.close()
                    raise
                location=response.headers.get('Location','')
                response.close()
            parsed=urllib.parse.urlsplit(location)
            if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password:
                raise RuntimeError('Invalid HTTPS artifact redirect')
            # Separate request/opener: no authenticated headers or token can
            # accompany the signed blob URL, including subsequent redirects.
            with urllib.request.urlopen(urllib.request.Request(location),timeout=min(30,deadline-time.monotonic())) as response, partial.open('wb') as output:
                digest=hashlib.sha256()
                downloaded=0
                while True:
                    if time.monotonic()>=deadline:
                        raise TimeoutError('Artifact download exceeded180seconds')
                    block=response.read(1024*1024)
                    if not block:
                        break
                    downloaded+=len(block)
                    if downloaded>EXPECTED_BYTES:
                        raise RuntimeError('Artifact exceeds the verified ZIP size')
                    digest.update(block)
                    output.write(block)
            if downloaded!=EXPECTED_BYTES or digest.hexdigest()!=EXPECTED:
                raise RuntimeError('Whole artifact ZIP SHA256 differs from independently verified artifact')
            partial.replace(DESTINATION)
            print('Verified official artifact12586985399 ZIP SHA256 '+EXPECTED)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt==2 or time.monotonic()+2>=deadline:
                raise
            time.sleep(2)


if __name__=='__main__':
    main()
