"""Small, self-contained font ONNX download kit; never includes uploaded files."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

PREDICT = '''import argparse,json
from pathlib import Path
from PIL import Image
from inference import RegionFontClassifier
from text_style import estimate_text_style
p=argparse.ArgumentParser(description="Predict font and size from a text-region image without OCR")
p.add_argument("image",type=Path)
a=p.parse_args()
with Image.open(a.image) as opened:image=opened.convert("RGB")
font=RegionFontClassifier(Path(__file__).resolve().parent).predict(image)
style=estimate_text_style(image,[],region_bbox=[0,0,*image.size])
style["font_size_px_estimate"]=font.get("font_size_px_estimate")
style["font_size_px_interval"]=None
style["size"]={"method":"region_neural_regression","unit":"source_image_px",
               "status":"estimated" if font.get("font_size_px_estimate") is not None else "unavailable"}
print(json.dumps({"font":font,"text_style":style,"ocr_performed":False},ensure_ascii=False,indent=2))
'''


def font_kit(engine):
    model=getattr(engine,'region_neural',None)
    if model is None:return None
    cached=getattr(engine,'_font_download',None)
    if cached is not None:return cached
    name=model.meta['model']['path']
    weights=(model.directory/name).read_bytes()
    if hashlib.sha256(weights).hexdigest()!=model.meta['model']['sha256']:
        raise ValueError('Model changed after serving startup')
    metadata={key:model.meta[key] for key in ('schema','algorithm','families','model','temperature','gates','max_size_relative_spread')}
    metadata['model_version']=engine.version
    requirements='numpy==2.2.6\npillow==12.3.0\nonnxruntime==1.23.2\n'
    readme=f'''# 字体识别 ONNX / Font recognition\n\n模型版本：{engine.version}\n\n直接输入一行或一个文字区域的图片，不需要 OCR、文字内容或字符切分。完整截图请先定位并裁出文字区域。\n\n```sh\npython -m venv .venv\n# macOS / Linux: source .venv/bin/activate\n# Windows: .venv\\Scripts\\activate\npip install -r requirements.txt\npython predict.py text-region.png\n```\n\nPython 3.10+。推理只需 ONNX Runtime、NumPy、Pillow，不需要 PyTorch。\n\n模型 ONNX 输入 tiles 为 float32 [N,1,64,256]；输出 logits [N,{len(model.families)}] 与 log_em_ratio [N]。必须使用 inference.py 中 preprocess_region 的比例保留、背景归一化和图块处理，并按 metadata.json 中的类别顺序和门槛解析。不要直接拉伸原图送入模型。\n\n输出 font.family 为字体候选，证据不足则为 null；候选分数不是实际准确率。font_size_px_estimate 为截图中的像素字号估计，不是 iOS pt；text_color_hex 为可见颜色。\n\n训练来源是 iOS Simulator 中受控原生页面的实际截图；未训练字体和混合字体仍可能误判。\n\n## Python API\n\n```python\nfrom pathlib import Path\nfrom PIL import Image\nfrom inference import RegionFontClassifier\nmodel = RegionFontClassifier(Path("."))\nresult = model.predict(Image.open("text-region.png").convert("RGB"))\nprint(result["family"], result["font_size_px_estimate"])\n```\n'''
    folder=Path(__file__).parent
    files={name:weights,'metadata.json':(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n').encode(),
           'inference.py':(folder/'region_font.py').read_bytes(),'text_style.py':(folder/'text_style.py').read_bytes(),
           'predict.py':PREDICT.encode(),'requirements.txt':requirements.encode(),'README.md':readme.encode()}
    files['SHA256.json']=(json.dumps({key:hashlib.sha256(value).hexdigest() for key,value in files.items()},indent=2)+'\n').encode()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for filename,data in files.items():
            info=zipfile.ZipInfo(filename,date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    data=stream.getvalue()
    info={'available':True,'format':'ONNX','version':engine.version,'families':model.families,
          'input_shape':[None,1,64,256],'download_url':'/api/models/font/download','bytes':len(data),
          'sha256':hashlib.sha256(data).hexdigest(),'ocr_required':False,
          'usage':{'install':'pip install -r requirements.txt','predict':'python predict.py text-region.png'}}
    engine._font_download=(info,data)
    return engine._font_download
