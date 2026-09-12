#!/usr/bin/env python3
"""Export, install and select checksum-verified offline model bundles."""
import argparse,json,os,re,shutil,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from flux_glyph.models import verify_bundle,load_active


def safe_version(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}',value):raise ValueError('Version must use letters/digits/dots/dashes/underscores')
    return value


def validate_runtime(directory):
    from flux_glyph.pipeline import FontPipeline
    engine=FontPipeline(directory,cache_characters=1)
    engine.bank.archive.close()
    if engine.latin_bank:engine.latin_bank.archive.close()


def select(root,relative):
    temporary=root/'ACTIVE.json.tmp';temporary.write_text(json.dumps({'path':relative},indent=2)+'\n');temporary.replace(root/'ACTIVE.json')


def activate(root,relative):
    directory=(root/relative).resolve()
    if not directory.is_relative_to(root.resolve()):raise ValueError('Model directory escapes root')
    verify_bundle(directory)
    validate_runtime(directory)
    select(root,relative)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model-root',type=Path,default=ROOT/'models')
    sub=parser.add_subparsers(dest='command',required=True)
    export=sub.add_parser('export');export.add_argument('--output',type=Path,required=True)
    install=sub.add_parser('install');install.add_argument('archive',type=Path);install.add_argument('--version',required=True)
    switch=sub.add_parser('activate');switch.add_argument('version',help='existing release name, or bundled for the original model')
    args=parser.parse_args();root=args.model_root.resolve()
    if args.command=='export':
        selected,version,manifest=load_active(root);args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists():raise ValueError('Refusing to overwrite an exported bundle')
        with zipfile.ZipFile(args.output,'w',zipfile.ZIP_STORED) as out:
            out.write(selected/'MANIFEST.json','MANIFEST.json')
            for row in manifest['files']:out.write(selected/row['path'],row['path'])
        print(json.dumps({'archive':str(args.output.resolve()),'version':version,'bytes':args.output.stat().st_size}))
    elif args.command=='install':
        version=safe_version(args.version);releases=root/'releases';releases.mkdir(exist_ok=True);target=releases/version
        if target.exists():raise ValueError('Release already exists; choose another version')
        temp=Path(tempfile.mkdtemp(prefix='.install-',dir=releases))
        try:
            with zipfile.ZipFile(args.archive) as archive:
                members=archive.infolist()
                if len(members)>10000 or sum(x.file_size for x in members)>256*1024*1024:raise ValueError('Model archive exceeds size/entry limit')
                seen=set()
                for member in members:
                    path=(temp/member.filename).resolve()
                    if not path.is_relative_to(temp.resolve()) or member.filename in seen or (member.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe model archive entry')
                    seen.add(member.filename)
                    if not member.is_dir():
                        path.parent.mkdir(parents=True,exist_ok=True)
                        with archive.open(member) as source,path.open('wb') as output:shutil.copyfileobj(source,output)
            verify_bundle(temp)
            validate_runtime(temp)
            temp.rename(target);select(root,'releases/'+version)
        finally:
            if temp.exists():shutil.rmtree(temp)
        print('Installed and selected '+version+'. Restart the API to load it.')
    else:
        relative='.' if args.version=='bundled' else 'releases/'+safe_version(args.version)
        activate(root,relative);print('Selected '+args.version+'. Restart the API to load it.')


if __name__=='__main__':main()
