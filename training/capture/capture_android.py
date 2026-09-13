#!/usr/bin/env python3
"""Build and capture an isolated Android Canvas application with font proofs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import unicodedata
import uuid
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent/'android'
PACKAGE = 'org.fluxglyph.androidcapture'
SCHEMA = 'flux-glyph-android-native-capture-v1'
_CACHE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def environment(args):
    env = dict(os.environ)
    env.update(JAVA_HOME=str(args.java.resolve()), ANDROID_SDK_ROOT=str(args.sdk.resolve()),
               ANDROID_HOME=str(args.sdk.resolve()), ANDROID_USER_HOME=str((args.work/'android-home').resolve()),
               ANDROID_AVD_HOME=str((args.work/'avd').resolve()), ANDROID_ADB_SERVER_PORT=str(args.adb_port))
    env['PATH'] = str(args.java/'bin') + os.pathsep + env['PATH']
    return env


def command(argv, args, **kwargs):
    return subprocess.run([str(value) for value in argv], env=environment(args), check=True, **kwargs)


def adb(args, *argv, **kwargs):
    return command([args.sdk/'platform-tools/adb', '-P', args.adb_port, '-s', f'emulator-{args.port}', *argv], args, **kwargs)


def source_bindings(scenes_path, scenes):
    paths = [Path(__file__), SOURCE/'CaptureActivity.java', SOURCE/'AndroidManifest.xml', Path(scenes_path)]
    for extra in scenes.get('bindings', {}):
        paths.append(Path(extra))
    paths.extend(Path(font['path']) for font in scenes['fonts'])
    return {str(path.resolve()): sha(path) for path in paths}


def validate_scenes(scenes):
    require(scenes.get('schema') == 'flux-glyph-android-scenes-v1' and scenes.get('canvas_px') == [1080, 2400], 'scene contract differs')
    require(all(Path(path).is_file() and sha(path)==digest for path,digest in scenes.get('bindings',{}).items()), 'scene source binding changed')
    fonts = {font['id']: font for font in scenes['fonts']}
    require(len(fonts) == len(scenes['fonts']) and len(scenes['families']) == 10 and scenes['families'][-1] == '__unknown__', 'font registry/order differs')
    pages, regions, texts, face_classes, unknown_sources, groups = set(), set(), {}, {}, {}, {}
    for font in fonts.values():
        require(re.fullmatch(r'[A-Za-z0-9_.-]+',font['id']) is not None,'unsafe font asset id')
        require(Path(font['path']).is_file() and sha(font['path']) == font['sha256'], 'font bytes changed')
        require(font['training_family'] in scenes['families'] and bool(font['postscript']), 'font source identity missing')
        identity=(font['sha256'],font.get('ttc_index',0))
        require(face_classes.setdefault(identity,font['training_family'])==font['training_family'],'one source face has conflicting training classes')
        if font['training_family']=='__unknown__':
            for identity in (('family',font['family']),('source',font['sha256'])):
                require(unknown_sources.setdefault(identity,font['split'])==font['split'],'unknown font family/source crosses partitions')
    for page in scenes['pages']:
        require(re.fullmatch(r'[A-Za-z0-9_.-]+',page['id']) is not None,'unsafe page id')
        require(page['id'] not in pages and page['split'] in ('train','calibration','test'), 'duplicate page or invalid split')
        pages.add(page['id'])
        require(groups.setdefault(page.get('content_group_id',page['id']),page['split'])==page['split'],'source content group crosses partitions')
        for row in page['regions']:
            require(row['id'] not in regions and row['font_id'] in fonts and bool(row['text']), 'duplicate/invalid region')
            regions.add(row['id'])
            font = fonts[row['font_id']]
            if font['training_family'] == '__unknown__':
                require(font['split'] == page['split'], 'negative font source crosses partitions')
            normalized=unicodedata.normalize('NFKC',row['text']).casefold()
            require(texts.setdefault(normalized, page['split']) == page['split'], 'text crosses partitions')
            require(isinstance(row['font_size_px'],(int,float)) and math.isfinite(row['font_size_px']) and 4<=row['font_size_px']<=512,'invalid native font size')
            require(re.fullmatch(r'#[0-9a-fA-F]{6}',row['color']) is not None and re.fullmatch(r'#[0-9a-fA-F]{6}',page['background']) is not None,'invalid native color')
            left, top, right, bottom = row['bbox']
            require(all(type(value) is int for value in row['bbox']),'native bbox must use integer pixels')
            require(0 <= left < right <= 1080 and 0 <= top < bottom <= 2400, 'region outside framebuffer')
    return fonts


def build(args):
    scenes = read(args.scenes)
    validate_scenes(scenes)
    require(not args.output.exists(), 'build output must be new')
    args.output.mkdir(parents=True)
    assets = args.output/'assets'; (assets/'fonts').mkdir(parents=True)
    packaged = json.loads(json.dumps(scenes))
    for font in packaged['fonts']:
        extension = Path(font['path']).suffix
        font['asset_path'] = 'fonts/'+font['id']+extension
        shutil.copyfile(font['path'], assets/font['asset_path'])
    dump(assets/'Scenes.json', packaged)
    classes, dex = args.output/'classes', args.output/'dex'
    classes.mkdir(); dex.mkdir()
    jar = args.sdk/'platforms/android-35/android.jar'; bt = args.sdk/'build-tools/35.0.1'
    command([args.java/'bin/javac', '--release','11','-classpath',jar,'-d',classes,SOURCE/'CaptureActivity.java'],args)
    command([bt/'d8','--lib',jar,'--min-api','31','--output',dex,*sorted(classes.rglob('*.class'))],args)
    unsigned = args.output/'unsigned.apk'; aligned = args.output/'aligned.apk'; apk=args.output/'capture.apk'
    # Stored font assets are mmap-able on Android. Compressed CJK assets allocate
    # a full direct heap copy per face and can exhaust the app before capture.
    command([bt/'aapt2','link','-o',unsigned,'--manifest',SOURCE/'AndroidManifest.xml','-I',jar,'-A',assets,
             '-0','ttf','-0','otf','-0','ttc'],args)
    with zipfile.ZipFile(unsigned,'a',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(dex/'classes.dex','classes.dex')
    command([bt/'zipalign','-f','4',unsigned,aligned],args)
    key = args.output/'capture-debug.jks'
    command([args.java/'bin/keytool','-genkeypair','-keystore',key,'-storepass','android','-keypass','android',
             '-alias','capture','-dname','CN=Flux Glyph Local Capture','-keyalg','RSA','-validity','3650'],args,
             stdout=subprocess.DEVNULL)
    command([bt/'apksigner','sign','--ks',key,'--ks-pass','pass:android','--key-pass','pass:android','--out',apk,aligned],args)
    command([bt/'apksigner','verify',apk],args)
    bindings = source_bindings(args.scenes,scenes)
    for name in ('javac','java'):
        bindings[str((args.java/'bin'/name).resolve())]=sha(args.java/'bin'/name)
    for tool in ('aapt2','d8','zipalign','apksigner'):
        bindings[str((bt/tool).resolve())]=sha(bt/tool)
    dump(args.output/'BUILD.json',{'schema':'flux-glyph-android-capture-build-v1','scenes':str(args.scenes.resolve()),
        'scenes_sha256':sha(args.scenes),'apk_sha256':sha(apk),'bindings':bindings})
    print(json.dumps({'apk':str(apk),'sha256':sha(apk),'pages':len(scenes['pages'])}),flush=True)


def boot(args):
    home=args.work/'avd'; folder=home/'flux_glyph_android_native_v1.avd'
    folder.mkdir(parents=True,exist_ok=True)
    require(not (folder/'hardware-qemu.ini.lock').exists(),'isolated AVD already running')
    config={'AvdId':'flux_glyph_android_native_v1','avd.ini.displayname':'Flux Glyph Android Native V1',
        'avd.ini.encoding':'UTF-8','abi.type':'arm64-v8a','hw.cpu.arch':'arm64','hw.cpu.ncore':'4',
        'hw.ramSize':'2048','hw.lcd.width':'1080','hw.lcd.height':'2400','hw.lcd.density':'440',
        'hw.mainKeys':'no','hw.keyboard':'yes','hw.gpu.enabled':'yes','hw.gpu.mode':'swiftshader',
        'image.sysdir.1':str((args.sdk/'system-images/android-35/google_apis/arm64-v8a').resolve())+'/',
        'tag.id':'google_apis','tag.display':'Google APIs','disk.dataPartition.size':'4G',
        'showDeviceFrame':'no','fastboot.forceColdBoot':'yes'}
    (folder/'config.ini').write_text(''.join(key+'='+value+'\n' for key,value in config.items()))
    (home/'flux_glyph_android_native_v1.ini').write_text('avd.ini.encoding=UTF-8\npath='+str(folder.resolve())+'\ntarget=android-35\n')
    (args.work/'android-home').mkdir(exist_ok=True)
    command([args.sdk/'platform-tools/adb','-P',args.adb_port,'start-server'],args)
    log=(args.work/'emulator.log').open('ab')
    process=subprocess.Popen([str(args.sdk/'emulator/emulator'),'-avd','flux_glyph_android_native_v1','-port',str(args.port),
        '-no-window','-no-boot-anim','-no-snapshot','-no-audio','-gpu','swiftshader','-camera-back','none','-camera-front','none'],
        env=environment(args),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        require(process.poll() is None,'emulator stopped; inspect emulator.log')
        status=subprocess.run([str(args.sdk/'platform-tools/adb'),'-P',str(args.adb_port),'-s',f'emulator-{args.port}',
            'shell','getprop','sys.boot_completed'],env=environment(args),capture_output=True)
        if status.returncode==0 and status.stdout.strip()==b'1':break
        time.sleep(1)
    else:raise RuntimeError('emulator boot timeout; inspect emulator.log')
    for name in ('window_animation_scale','transition_animation_scale','animator_duration_scale'):
        adb(args,'shell','settings','put','global',name,'0',stdout=subprocess.DEVNULL)
    adb(args,'shell','settings','put','secure','immersive_mode_confirmations','confirmed',stdout=subprocess.DEVNULL)
    adb(args,'shell','input','keyevent','82',stdout=subprocess.DEVNULL)
    print(json.dumps({'pid':process.pid,'serial':f'emulator-{args.port}','adb_port':args.adb_port,'log':str(args.work/'emulator.log')}),flush=True)


def capture(args):
    built=read(args.build/'BUILD.json'); scenes_path=Path(built['scenes']); scenes=read(scenes_path)
    fonts=validate_scenes(scenes)
    require(sha(scenes_path)==built['scenes_sha256'] and sha(args.build/'capture.apk')==built['apk_sha256'],'build source changed')
    require(all(sha(path)==digest for path,digest in built['bindings'].items()),'bound capture code/tool/font changed')
    require(not args.output.exists(),'capture output must be new')
    args.output.mkdir(parents=True); (args.output/'screenshots').mkdir(); (args.output/'proofs').mkdir()
    shutil.copyfile(scenes_path,args.output/'Scenes.json')
    if adb(args,'shell','pm','path',PACKAGE,capture_output=True).stdout.strip():
        adb(args,'uninstall',PACKAGE,stdout=subprocess.DEVNULL)
    adb(args,'install','-t',args.build/'capture.apk',stdout=subprocess.DEVNULL)
    adb(args,'shell','settings','put','secure','immersive_mode_confirmations','confirmed',stdout=subprocess.DEVNULL)
    adb(args,'shell','am','force-stop',PACKAGE,stdout=subprocess.DEVNULL)
    labels=[]; files=[]
    selected=[(i,page) for i,page in enumerate(scenes['pages']) if not args.split or page['split']==args.split]
    if args.limit is not None:selected=selected[:args.limit]
    for number,(index,page) in enumerate(selected):
        token=uuid.uuid4().hex
        adb(args,'shell','am','start','-W','-n',PACKAGE+'/.CaptureActivity','--ei','page_index',str(index),
            '--es','capture_token',token,stdout=subprocess.DEVNULL)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            ready=subprocess.run([str(args.sdk/'platform-tools/adb'),'-P',str(args.adb_port),'-s',f'emulator-{args.port}',
                'shell','run-as',PACKAGE,'cat','files/ready.json'],env=environment(args),capture_output=True)
            try:proof=json.loads(ready.stdout)
            except (json.JSONDecodeError,UnicodeDecodeError):proof={}
            if 'error' in proof and proof.get('capture_token') in (None,token):raise RuntimeError(proof['error'])
            if proof.get('capture_token')==token:break
            time.sleep(.05)
        else:raise RuntimeError('capture readiness timeout: '+page['id'])
        require(proof['page_id']==page['id'] and proof['split']==page['split'] and len(proof['regions'])==len(page['regions']), 'wrong native page')
        require(all(row['font_verified'] is True for row in proof['regions']), 'fallback/missing glyph/clipped native region: '+page['id']+' '+str([r['id'] for r in proof['regions'] if not r['font_verified']]))
        screenshot=adb(args,'exec-out','screencap','-p',capture_output=True).stdout
        image=Image.open(io.BytesIO(screenshot));require(image.size==tuple(scenes['canvas_px']),'framebuffer dimensions differ')
        image_path='screenshots/'+page['id']+'.png';proof_path='proofs/'+page['id']+'.json'
        (args.output/image_path).write_bytes(screenshot);dump(args.output/proof_path,proof)
        files.extend({'path':path,'sha256':sha(args.output/path)} for path in (image_path,proof_path))
        for requested,actual in zip(page['regions'],proof['regions']):
            font=fonts[requested['font_id']]
            require(actual['id']==requested['id'] and actual['text']==requested['text'],'native region order differs')
            labels.append({'schema':'flux-glyph-android-native-region-v1','split':page['split'],'source_id':'android:'+page['id'],
                'page_id':page['id'],'content_group_id':page.get('content_group_id',page['id']),'region_id':requested['id'],
                'image':image_path,'image_sha256':files[-2]['sha256'],'image_width':image.width,'image_height':image.height,
                'bbox':actual['bbox'],'ink_bbox':actual['ink_bbox'],'text':requested['text'],'script':requested['script'],
                'font_id':font['id'],'font_family':font['family'],'training_family':font['training_family'],
                'font_face':font['postscript'],'font_file_sha256':font['sha256'],'ttc_index':font.get('ttc_index',0),
                'font_size_screen_px':actual['font_size_px'],'color':actual['color'],'background':page['background'],
                'native_font_verified':True,'native_rendering':'Android Canvas + TextRunShaper','proof':proof_path})
        if (number+1)%10==0 or number+1==len(selected):print(json.dumps({'pages':number+1,'total':len(selected),'regions':len(labels),'split':page['split']}),flush=True)
    labels_path=args.output/'labels.jsonl'
    labels_path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in labels))
    bindings=dict(built['bindings']);bindings[str((args.build/'BUILD.json').resolve())]=sha(args.build/'BUILD.json')
    bindings[str((args.build/'capture.apk').resolve())]=sha(args.build/'capture.apk')
    manifest={'schema':SCHEMA,'families':scenes['families'],'fonts':scenes['fonts'],'bindings':bindings,
        'scenes':{'path':'Scenes.json','sha256':sha(args.output/'Scenes.json')},'labels':{'path':'labels.jsonl','sha256':sha(labels_path)},
        'files':files,'pages':len(selected),'regions':len(labels),'split_counts':dict(Counter(row['split'] for row in labels)),
        'device':{'serial':f'emulator-{args.port}','build_fingerprint':proof['build_fingerprint'],'sdk_int':proof['sdk_int']},
        'rendering':'Native Android TextRunShaper and Canvas.drawGlyphs, framebuffer captured using adb screencap; no host font rendering.',
        'all_native_fonts_verified':True,'smoke_only':args.limit is not None,'user_images_used':False}
    require(all(sha(path)==digest for path,digest in bindings.items()),'capture input changed during run')
    dump(args.output/'CAPTURE_MANIFEST.json',manifest)
    loaded=load_capture(args.output)
    for row in loaded['rows']:verify_region(args.output,row)
    print(json.dumps({'manifest':str(args.output/'CAPTURE_MANIFEST.json'),'sha256':sha(args.output/'CAPTURE_MANIFEST.json'),'rows':len(labels)}),flush=True)


def load_capture(root):
    """Load audited metadata only; never decode any image or model TEST tiles."""
    root=Path(root).resolve();manifest=read(root/'CAPTURE_MANIFEST.json')
    require(manifest.get('schema')==SCHEMA and manifest.get('all_native_fonts_verified') is True,'capture manifest contract differs')
    for field in ('scenes','labels'):
        descriptor=manifest[field];require(sha(root/descriptor['path'])==descriptor['sha256'],'capture metadata changed')
    require(all(Path(path).is_file() and sha(path)==digest for path,digest in manifest['bindings'].items()),'capture source binding changed')
    scenes=read(root/manifest['scenes']['path']);fonts=validate_scenes(scenes)
    require(scenes['families']==manifest['families'] and scenes['fonts']==manifest['fonts'],'capture class/source order differs')
    rows=[json.loads(line) for line in (root/manifest['labels']['path']).read_text().splitlines()]
    require(len(rows)==manifest['regions'] and len({row['region_id'] for row in rows})==len(rows),'capture row identities differ')
    index={row['id']:(page,row) for page in scenes['pages'] for row in page['regions']}
    filemap={item['path']:item['sha256'] for item in manifest['files']}
    require(len(filemap)==len(manifest['files']),'duplicate captured file')
    require(set(filemap)=={row[key] for row in rows for key in ('image','proof')},'unexpected or absent captured file')
    require(dict(Counter(row['split'] for row in rows))==manifest['split_counts']
            and len({row['page_id'] for row in rows})==manifest['pages'],'capture partition counts differ')
    for row in rows:
        require(row['region_id'] in index,'label absent from scene')
        page,requested=index[row['region_id']]
        verify_label_metadata(row,page,requested,fonts[requested['font_id']],scenes['canvas_px'])
        require(row['image'] in filemap and row['proof'] in filemap and row['image_sha256']==filemap[row['image']],'label file binding differs')
    _CACHE[str(root)]={'manifest':manifest,'rows':{r['region_id']:r for r in rows},'index':index,'fonts':fonts,'files':filemap,'proofs':{}}
    bindings=dict(manifest['bindings'])
    for name in ('CAPTURE_MANIFEST.json',manifest['scenes']['path'],manifest['labels']['path']):bindings[str(root/name)]=sha(root/name)
    return {'families':manifest['families'],'rows':rows,'manifest_sha256':sha(root/'CAPTURE_MANIFEST.json'),'bindings':bindings}


def verify_label_metadata(row,page,requested,font,canvas):
    expected={'schema':'flux-glyph-android-native-region-v1','split':page['split'],'source_id':'android:'+page['id'],
        'page_id':page['id'],'content_group_id':page.get('content_group_id',page['id']),'region_id':requested['id'],
        'image':'screenshots/'+page['id']+'.png','proof':'proofs/'+page['id']+'.json',
        'image_width':canvas[0],'image_height':canvas[1],'bbox':requested['bbox'],'text':requested['text'],'script':requested['script'],
        'font_id':font['id'],'font_family':font['family'],'training_family':font['training_family'],'font_face':font['postscript'],
        'font_file_sha256':font['sha256'],'ttc_index':font.get('ttc_index',0),'font_size_screen_px':requested['font_size_px'],
        'background':page['background'],'native_font_verified':True,'native_rendering':'Android Canvas + TextRunShaper'}
    require(all(row.get(key)==value for key,value in expected.items())
        and row.get('color','').upper()==requested['color'].upper(),'label metadata differs from bound native scene/font')


def verify_region(root,row):
    """Verify scene, label and native glyph proof; only this row's image is hashed."""
    root=Path(root).resolve()
    if str(root) not in _CACHE:load_capture(root)
    cache=_CACHE[str(root)]; require(cache['rows'].get(row['region_id'])==row,'row differs from signed labels')
    page,requested=cache['index'][row['region_id']];font=cache['fonts'][requested['font_id']]
    proof_path=row['proof']
    require(sha(root/proof_path)==cache['files'][proof_path],'native proof changed')
    require(sha(root/row['image'])==cache['files'][row['image']]==row['image_sha256'],'native screenshot changed')
    if proof_path not in cache['proofs']:
        cache['proofs'][proof_path]=read(root/proof_path)
    proof=cache['proofs'][proof_path]
    require(proof['canvas_px']==[row['image_width'],row['image_height']]
        and proof['build_fingerprint']==cache['manifest']['device']['build_fingerprint']
        and proof['sdk_int']==cache['manifest']['device']['sdk_int'],'native device/viewport differs')
    actual=next(item for item in proof['regions'] if item['id']==row['region_id'])
    require(row['split']==page['split']==proof['split'] and row['page_id']==page['id']==proof['page_id'],'native source split differs')
    require(row['source_id']=='android:'+page['id'] and row['text']==requested['text']==actual['text']
        and row['font_id']==requested['font_id']==actual['font_id'],'native label identity differs')
    require(row['font_family']==font['family'] and row['training_family']==font['training_family']
        and row['font_face']==font['postscript'] and row['font_file_sha256']==font['sha256'],'native font label differs')
    require(row['bbox']==requested['bbox']==actual['bbox'] and row['ink_bbox']==actual['ink_bbox']
        and row['font_size_screen_px']==requested['font_size_px']==actual['font_size_px']
        and row['color'].upper()==requested['color'].upper()==actual['color'].upper(),'native pixel geometry/style differs')
    require(row['native_font_verified'] is True and actual['font_verified'] is True and actual['glyphs'],'native font proof missing')
    for glyph in actual['glyphs']:
        evidence=glyph['font']
        require(glyph['font_verified'] is True and glyph['glyph_id']!=0 and evidence['sha256']==font['sha256']
            and evidence['postscript']==font['postscript'] and evidence['ttc_index']==font.get('ttc_index',0),
            'native fallback or missing glyph detected')


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--sdk',type=Path,default=ROOT/'artifacts/android-sdk')
    result.add_argument('--java',type=Path,default=Path('/Applications/PyCharm.app/Contents/jbr/Contents/Home'))
    result.add_argument('--work',type=Path,default=ROOT/'artifacts/android-native-v1')
    result.add_argument('--port',type=int,default=5580);result.add_argument('--adb-port',type=int,default=5039)
    commands=result.add_subparsers(dest='action',required=True)
    commands.add_parser('boot')
    build_parser=commands.add_parser('build');build_parser.add_argument('--scenes',type=Path,required=True);build_parser.add_argument('--output',type=Path,required=True)
    capture_parser=commands.add_parser('capture');capture_parser.add_argument('--build',type=Path,required=True);capture_parser.add_argument('--output',type=Path,required=True)
    capture_parser.add_argument('--split',choices=['train','calibration','test']);capture_parser.add_argument('--limit',type=int)
    return result


if __name__=='__main__':
    arguments=parser().parse_args()
    {'build':build,'boot':boot,'capture':capture}[arguments.action](arguments)
