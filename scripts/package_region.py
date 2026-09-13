"""Package only the text detector and OCR-free region font/size CNN."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from flux_glyph.models import load_active,verify_bundle
from flux_glyph.region_font import RegionFontClassifier
from flux_glyph.android_font import AndroidFontClassifier
from flux_glyph.unified_font import UnifiedFontClassifier
from model_release import safe_version,validate_runtime
from package_models import write_models_manifest


def package(base,region,output,version):
    version=safe_version(version)
    selected,base_version,_=load_active(base)
    metadata=json.loads((Path(region)/'metadata.json').read_text())
    classifier=(UnifiedFontClassifier(region) if metadata.get('schema')=='flux-glyph-unified-region-font-v1'
                else AndroidFontClassifier(region) if metadata.get('schema')=='flux-glyph-android-region-font-v1'
                else RegionFontClassifier(region))
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Choose a new bundle output directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.region-package-',dir=output.parent))
    try:
        for relative in ('pp/onnx/paddle_ocr_det.onnx','pp/paddle_ocr_delivery.contract.json'):
            path=temp/relative;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(selected/relative,path)
        (temp/'region_neural').mkdir()
        filenames=['metadata.json',classifier.meta['model']['path']]
        if getattr(classifier,'rejection_meta',None) is not None:filenames.append(classifier.rejection_meta['model']['path'])
        if getattr(classifier,'verifier_meta',None) is not None:filenames.append(classifier.verifier_meta['model']['path'])
        for name in filenames:
            shutil.copyfile(Path(region)/name,temp/'region_neural'/name)
        manifest=write_models_manifest(temp)
        manifest.update(version=version,base_version=base_version,font_method='region_neural_network',ocr_performed=False,
                        font_mode=getattr(classifier,'font_mode','ios'),
                        source_note='Detector and trained region font/size CNN; no text recognizer, dictionary, glyph segmenter or reference bank.')
        (temp/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        verify_bundle(temp);validate_runtime(temp)
        temp.rename(output)
    finally:
        if temp.exists():shutil.rmtree(temp)
    return {'directory':str(output),'version':version,'files':len(manifest['files']),'bytes':sum(f['bytes'] for f in manifest['files'])}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'models')
    p.add_argument('--region',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--version',default='r16-ios-region-v1')
    a=p.parse_args();print(json.dumps(package(a.base,a.region,a.output,a.version),indent=2))
