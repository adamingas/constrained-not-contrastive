"""Download pinned public evaluation sources. Preserve original dataset licenses."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import requests
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/long_context';DEST.mkdir(parents=True,exist_ok=True)
FILES={
 'quality.dev.jsonl':('https://raw.githubusercontent.com/nyu-mll/quality/f84977c40dbfef70c9cab48037b7becfc8e45f73/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev','99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef'),
 'longbench_v2.json':('https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json','15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2')}

def fetch(item):
 name,(url,sha)=item;path=DEST/name
 if not path.exists():
  with requests.get(url,stream=True,timeout=180) as r:
   r.raise_for_status()
   with path.open('wb') as f:
    for chunk in r.iter_content(1024*1024):f.write(chunk)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,name
 print('verified',name,flush=True)

if __name__=='__main__':
 with ThreadPoolExecutor(2) as ex:list(ex.map(fetch,FILES.items()))
 path=DEST/'contract_test.json'
 if not path.exists():
  url='https://raw.githubusercontent.com/stanfordnlp/contract-nli/eced6528dd3c1d14d73f9a87df8f7bdbc03126f9/resources/contract-nli.zip'
  r=requests.get(url,timeout=180);r.raise_for_status();z=zipfile.ZipFile(io.BytesIO(r.content))
  path.write_bytes(z.read('contract-nli/test.json'))
  for name in ['LICENSE','TERMS','README.md']:(DEST/('contract_'+name)).write_bytes(z.read('contract-nli/'+name))
 assert hashlib.sha256(path.read_bytes()).hexdigest()=='460267b56052a2dc5aead98eb35eadef9e6734d5723d37b4a9790e410f812387'
 print('verified contract_test.json')
