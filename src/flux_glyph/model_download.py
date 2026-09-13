"""Small, self-contained font ONNX download kit; never includes uploaded files."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
from .region_font import rejection_metadata, verifier_metadata
from .android_font import android_metadata
from .unified_font import unified_metadata

PREDICT = '''import argparse,json
from pathlib import Path
from PIL import Image
from inference import RegionFontClassifier
from text_style import estimate_text_style
p=argparse.ArgumentParser(description="Predict font and size from a text-region image without OCR")
p.add_argument("image",type=Path)
a=p.parse_args()
with Image.open(a.image) as opened:image=opened.convert("RGB")
model=RegionFontClassifier(Path(__file__).resolve().parent)
font=model.predict(image)
style=estimate_text_style(image,[],region_bbox=[0,0,*image.size])
style["font_size_px_estimate"]=font.get("font_size_px_estimate")
style["font_size_px_interval"]=None
style["size"]={"method":"region_neural_regression","unit":"source_image_px",
               "status":"estimated" if font.get("font_size_px_estimate") is not None else "unavailable"}
result={"font":font,"text_style":style,"ocr_performed":False}
if model.meta.get("release_tier")=="experimental":
    result["model_release_tier"]="experimental"
    if type(model.meta.get("stable_validation_passed")) is bool:
        result["stable_validation_passed"]=model.meta["stable_validation_passed"]
print(json.dumps(result,ensure_ascii=False,indent=2))
'''


def release_metadata(source):
    """Copy explicitly published release evidence, without inventing a tier."""
    result={key:source[key] for key in ('release_tier','stable_validation_passed','test_passed','validation') if key in source}
    if 'release_tier' in result and (not isinstance(result['release_tier'],str) or not result['release_tier'] or len(result['release_tier'])>64):
        raise ValueError('Invalid public model release tier')
    if any(type(result[key]) is not bool for key in ('stable_validation_passed','test_passed') if key in result):
        raise ValueError('Invalid public model validation flag')
    if 'validation' in result and not isinstance(result['validation'],dict):
        raise ValueError('Invalid public model validation evidence')
    pending=[(result,0)];items=0
    while pending:
        value,depth=pending.pop();items+=1
        if depth>16 or items>4096:raise ValueError('Public model validation evidence exceeds bounds')
        if type(value) is dict:
            if not all(type(key) is str for key in value):raise ValueError('Public model validation keys must be strings')
            pending.extend((item,depth+1) for item in value.values())
        elif type(value) is list:pending.extend((item,depth+1) for item in value)
        elif type(value) not in (str,int,float,bool,type(None)):
            raise ValueError('Public model validation evidence is not JSON data')
        elif type(value) is int and abs(value)>2**53-1:
            raise ValueError('Public model validation integer exceeds JSON-safe precision')
    encoded=json.dumps(result,ensure_ascii=False,allow_nan=False)
    if len(encoded.encode())>65536:raise ValueError('Public model validation evidence exceeds bounds')
    return json.loads(encoded)


def font_kit(engine):
    model=getattr(engine,'region_neural',None)
    if model is None:return None
    cached=getattr(engine,'_font_download',None)
    if cached is not None:return cached
    name=model.meta['model']['path']
    weights=(model.directory/name).read_bytes()
    if hashlib.sha256(weights).hexdigest()!=model.meta['model']['sha256']:
        raise ValueError('Model changed after serving startup')
    android=model.meta.get('schema')=='flux-glyph-android-region-font-v1'
    unified=model.meta.get('schema')=='flux-glyph-unified-region-font-v1'
    single=android or unified
    metadata=(unified_metadata(model.meta) if unified else android_metadata(model.meta) if android else
              {key:model.meta[key] for key in ('schema','algorithm','families','model','temperature','gates','max_size_relative_spread')})
    metadata['font_mode']='unified' if unified else 'android' if android else 'ios'
    metadata['model_version']=engine.version
    metadata['font_label_groups']=model.meta.get('font_label_groups',{})
    metadata['font_sources']=model.meta.get('font_sources',{})
    release=release_metadata(model.meta)
    metadata.update(release)
    rejection=None if single else rejection_metadata(model.meta)
    rejection_bytes=None
    if rejection is not None:
        rejection_path=model.directory/rejection['model']['path']
        if (not rejection_path.resolve().is_relative_to(model.directory.resolve())
                or not rejection_path.is_file() or rejection_path.stat().st_size>64*1024*1024):
            raise ValueError('Rejection model is missing or exceeds bounds')
        rejection_bytes=rejection_path.read_bytes()
        if hashlib.sha256(rejection_bytes).hexdigest()!=rejection['model']['sha256']:
            raise ValueError('Rejection model changed after serving startup')
        metadata['rejection']=rejection
    verifier=None if single else verifier_metadata(model.meta)
    verifier_bytes=None
    if verifier is not None:
        verifier_path=model.directory/verifier['model']['path']
        if (not verifier_path.resolve().is_relative_to(model.directory.resolve())
                or not verifier_path.is_file() or verifier_path.stat().st_size>64*1024*1024):
            raise ValueError('Verifier model is missing or exceeds bounds')
        verifier_bytes=verifier_path.read_bytes()
        if hashlib.sha256(verifier_bytes).hexdigest()!=verifier['model']['sha256']:
            raise ValueError('Verifier model changed after serving startup')
        metadata['verifier']=verifier
    requirements='numpy==2.2.6\npillow==12.3.0\nonnxruntime==1.23.2\n'
    readme=f'''# 字体识别 ONNX / Font recognition\n\n模型版本：{engine.version}\n\n直接输入一行或一个文字区域的图片，不需要 OCR、文字内容或字符切分。完整截图请先定位并裁出文字区域。\n\n```sh\npython -m venv .venv\n# macOS / Linux: source .venv/bin/activate\n# Windows: .venv\\Scripts\\activate\npip install -r requirements.txt\npython predict.py text-region.png\n```\n\nPython 3.10+。推理只需 ONNX Runtime、NumPy、Pillow，不需要 PyTorch。\n\n模型 ONNX 输入 tiles 为 float32 [N,1,64,256]；输出 logits [N,{len(model.families)}] 与 log_em_ratio [N]。必须使用 inference.py 中 preprocess_region 的比例保留、背景归一化和图块处理，并按 metadata.json 中的类别顺序和门槛解析。不要直接拉伸原图送入模型。\n\n输出 font.family 为字体候选，证据不足则为 null；候选分数不是实际准确率。font_size_px_estimate 为截图中的像素字号估计，不是 iOS pt；text_color_hex 为可见颜色。\n\n训练来源是 iOS Simulator 中受控原生页面的实际截图；未训练字体和混合字体仍可能误判。\n\n## Python API\n\n```python\nfrom pathlib import Path\nfrom PIL import Image\nfrom inference import RegionFontClassifier\nmodel = RegionFontClassifier(Path("."))\nresult = model.predict(Image.open("text-region.png").convert("RGB"))\nprint(result["family"], result["font_size_px_estimate"])\n```\n'''
    if unified:
        readme=readme.replace('iOS Simulator 中受控原生页面','iOS Simulator 与 Android Emulator 中受控原生页面')
        readme=readme.replace('不是 iOS pt','不是 iOS pt 或 Android sp／dp')
        readme=readme.replace(f'logits [N,{len(model.families)}]',f'logits [N,{len(model.meta["families"])}]')
        readme+=f'\n本统一模型以同一个字体 CNN 联合识别 {len(model.families)} 类已覆盖字体，无需选择 iOS／Android，也不据此判断设备系统。全部命名类别与 __unknown__ 在同一个 softmax 中竞争；未知类胜出时清空字体名称、命名候选和字号。不能删除未知类后重新归一化。类别顺序以 metadata.json 为准。font_label_groups 表示共同识别组，不代表能精确区分相同字形的别名。\n'
        if release.get('release_tier')=='experimental':
            notice=f'统一字体实验模型，覆盖{len(model.families)}类字体；范围外字体仍可能误命名，分数不是准确率。'
            readme=readme.replace(f'模型版本：{engine.version}\n',f'模型版本：{engine.version}\n\n**{notice}**\n',1)
            readme+='\n发布层级及实际验收结果原样保存在 metadata.json 的 release_tier、stable_validation_passed、test_passed 和 validation 字段（若该版本声明）。实验版标识本身不代表稳定验收通过。\n'
            if release.get('stable_validation_passed') is False:readme+='\n本版本尚未通过稳定版验收。\n'
    elif android:
        readme=readme.replace('iOS Simulator','Android Emulator').replace('不是 iOS pt','不是 Android sp 或 dp').replace(f'logits [N,{len(model.families)}]',f'logits [N,{len(model.meta["families"])}]')
        readme += '\n这是用户指定的安卓字体识别范围，使用独立训练的字体 CNN，不是手机系统鉴定。模型保留 __unknown__ 拒识类；它胜出时不公开字体名称或字号。命名类别的分数仍保留完整十类竞争，不能删除未知类后重新归一化。font_label_groups 中的名称属于共同识别组，不代表可以从截图精确区分各个别名或地区版本。\n'
        if release.get('release_tier')=='experimental':
            notice=f'Android 实验候选模型，覆盖{len(model.families)}类字体，尚未通过稳定版验收；范围外字体仍可能误命名，分数不是准确率。'
            readme=readme.replace(f'模型版本：{engine.version}\n',f'模型版本：{engine.version}\n\n**{notice}**\n\nExperimental Android preview: stable validation has not passed. Fonts outside the supported set may still receive incorrect names; model scores are not accuracy.\n',1)
            validation=release.get('validation',{})
            if validation.get('training_font_faces')==33 and validation.get('all_capture_font_faces')==35:
                readme+='\n本次训练使用 33 个字体 face，全部采集共 35 个字体 face；这些来源包含九类命名字体和范围外训练／验证字体。\n'
            if validation.get('onnx_parity_passed') is True:
                readme+='\nCPU ONNX 完整数值、字体及字号决策校验已通过。这是执行一致性验证，不代表识别准确率或稳定版 TEST 验收通过。\n'
            if release.get('test_passed') is False:
                readme+='\n稳定版 TEST 验收未通过。完整公开指标与测试范围保存在 metadata.json 的 validation 字段；页面说明见 /docs#android-preview。\n'
    else:
        readme += '\n字体列表既包含系统字体，也包含应用自带字体；Alipay Number 是应用字体，不是 iOS 系统字体。字体来源记录在 metadata.json 的 font_sources 中（有记录的版本）。\n'
    if not single and metadata['font_label_groups']:
        readme += '\n本版 PingFang 表示苹方字体族，覆盖 SC／TC／HK 的原生简繁体样本。多个地区字体存在相同字形，图片没有足够信息时不宣称能区分地区版本；对应原生名称见 font_label_groups。\n'
    if rejection is not None:
        readme += '\n本包还包含独立的未知字体拒识神经网络。使用相同区域图块，输出 known_logits [N,2]，类别顺序 unknown、known；先按 metadata.json 中的温度与门槛判断覆盖情况。拒识时不输出字体名称、命名候选或字号，仍可测量原图颜色。类别相对分数与覆盖检测分数均不是实测准确率。两个 ONNX 必须一起保留，不得绕过拒识模型。\n'
    if verifier is not None:
        readme += '\n本版另含独立字体复核网络，输入同一组图块，输出 logits 与 log_em_ratio；只用它的 logits 复核字体，字号仍来自主网络。三个 ONNX 必须一起保留。通过未知字体检查后，两网赢家相同且各自达到门槛才输出字体与字号。分歧时 font.family 与字号为 null，font.candidates 和 font.verifier.candidates 展示双方类别相对分数。复核网络中的额外竞争类（如 Roboto）达到门槛时仅用于拒绝主网络未覆盖的字体，不表示主模型支持命名这些额外字体。\n'
    folder=Path(__file__).parent
    files={name:weights,'metadata.json':(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n').encode(),
           'inference.py':(folder/('unified_font.py' if unified else 'android_font.py' if android else 'region_font.py')).read_bytes(),'text_style.py':(folder/'text_style.py').read_bytes(),
           'predict.py':PREDICT.encode(),'requirements.txt':requirements.encode(),'README.md':readme.encode()}
    if rejection is not None:
        files[rejection['model']['path']]=rejection_bytes
    if verifier is not None:
        files[verifier['model']['path']]=verifier_bytes
    if single:files['region_font.py']=(folder/'region_font.py').read_bytes()
    files['SHA256.json']=(json.dumps({key:hashlib.sha256(value).hexdigest() for key,value in files.items()},indent=2)+'\n').encode()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for filename,data in files.items():
            info=zipfile.ZipInfo(filename,date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    data=stream.getvalue()
    info={'available':True,'format':'ONNX','version':engine.version,'families':model.families,
          'font_label_groups':metadata['font_label_groups'],'font_sources':metadata['font_sources'],
          'input_shape':[None,1,64,256],'download_url':'/api/models/font/download','bytes':len(data),
          'font_mode':'unified' if unified else 'android' if android else 'ios',
          'sha256':hashlib.sha256(data).hexdigest(),'ocr_required':False,
          'unknown_font_rejection':single or rejection is not None,
          'font_consensus_verification':verifier is not None,
          'usage':{'install':'pip install -r requirements.txt','predict':'python predict.py text-region.png'}}
    info.update(release_metadata(metadata))
    engine._font_download=(info,data)
    return engine._font_download
