#!/usr/bin/env python3
"""Build a deployable source/model ZIP without uploads, caches or local secrets."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from flux_glyph.models import load_active

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/flux-glyph-deploy-v1.zip')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Choose a new output file; existing release will not be overwritten')
    selected,version,manifest=load_active(ROOT/'models')
    files={name:ROOT/name for name in ('Dockerfile','compose.yaml','requirements.txt','requirements.lock','README.md','.env.example','.dockerignore','.gitignore','.gitattributes')}
    for folder in ('src','web','assets','scripts','tests','docs'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and not any(p.startswith('.') or p=='__pycache__' for p in path.relative_to(ROOT).parts) and path.suffix!='.pyc':
                files[path.relative_to(ROOT).as_posix()]=path
    files['models/MANIFEST.json']=selected/'MANIFEST.json'
    for item in manifest['files']:files['models/'+item['path']]=selected/item['path']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in sorted(files.items()):archive.write(path,'flux-glyph/'+name)
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    report={'archive':args.output.name,'bytes':args.output.stat().st_size,'sha256':digest,'model_version':version,'file_count':len(files),
            'contents':'source, website, models, deployment files, tests, validation reports; no uploads, .env, venv or training source fonts'}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
