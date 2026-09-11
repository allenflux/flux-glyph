"""One-worker FastAPI service for small CPU servers."""
from __future__ import annotations
import asyncio
import hmac
import io
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from python_multipart.exceptions import MultipartParseError
from PIL import Image,UnidentifiedImageError
from .pipeline import FontPipeline,ROOT,save_json

MAX_BYTES=int(os.getenv('FLUX_MAX_UPLOAD_MB','8'))*1024*1024
MAX_PIXELS=int(os.getenv('FLUX_MAX_PIXELS','12000000'))
MAX_QUEUE=int(os.getenv('FLUX_QUEUE_SIZE','8'))
MAX_JOBS=int(os.getenv('FLUX_MAX_SAVED_JOBS','100'))
TTL_HOURS=float(os.getenv('FLUX_RETENTION_HOURS','168'))
DATA=Path(os.getenv('FLUX_DATA_DIR',str(ROOT/'data'))).resolve()
TOKEN=os.getenv('FLUX_API_TOKEN','')
Image.MAX_IMAGE_PIXELS=MAX_PIXELS
logger=logging.getLogger('flux_glyph')


def validate_image(payload):
    if not payload or len(payload)>MAX_BYTES:raise ValueError('图片为空或超过上传大小限制。')
    with Image.open(io.BytesIO(payload)) as image:
        if image.format not in ('PNG','JPEG','WEBP'):raise ValueError('支持 PNG、JPG、WebP 图片。')
        if image.width*image.height>MAX_PIXELS or min(image.size)<8 or max(image.size)>16000:raise ValueError('图片尺寸超出处理范围。')
        if getattr(image,'n_frames',1)>1:raise ValueError('请上传单张静态图片。')
        image.verify()


def public_result(identifier,result):
    prefix=f'/assets/{identifier}/'
    regions=[]
    for reg in result['regions']:
        glyphs=[{k:g.get(k) for k in ('character','index','status','reason','family_candidate','candidates','source_bbox','source_rotation_degrees')} |
                ({'crop_url':prefix+g['crop_file']} if g.get('crop_file') else {}) for g in reg['glyphs']]
        regions.append({k:reg.get(k) for k in ('id','quad','text','font','detector_bbox','source_bbox','ocr_confidence')} |
                       {'glyphs':glyphs,'crop_url':prefix+reg['crop_file']})
    return {k:result[k] for k in ('id','width','height','summary','timing_seconds','model_version','source_sha256','font_scope','font_identity_verified','device_inference_performed')} | {
        'regions':regions,'image_url':prefix+'original.png','annotated_preview_url':prefix+'annotated.png',
        'annotated_image_url':f'/api/jobs/{identifier}/image','download_json_url':f'/api/jobs/{identifier}/json'}


class Jobs:
    def __init__(self):
        DATA.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='glyph')
        self.capacity=threading.BoundedSemaphore(MAX_QUEUE+1);self.active=set();self.jobs={}
        self.engine=FontPipeline(Path(os.getenv('FLUX_MODEL_DIR',str(ROOT/'models'))),cache_characters=int(os.getenv('FLUX_FONT_CACHE_CHARACTERS','32')),max_regions=int(os.getenv('FLUX_MAX_REGIONS','200')))
        for path in DATA.glob('*/job.json'):
            if not re.fullmatch('[a-f0-9]{32}',path.parent.name):continue
            try:
                job=json.loads(path.read_text())
                if job['status'] in ('queued','running'):
                    job.update(status='error',stage='服务已重启',error='任务中断，请重新提交图片。',error_code='service_restarted',
                               progress={'stage_code':'error','percent':None,'current':None,'total':None},finished_at=time.time())
                    save_json(path,job)
                self.jobs[path.parent.name]=job
            except (ValueError,KeyError):continue
        self.cleanup()

    def cleanup(self):
        with self.lock:
            # Include orphaned directories (e.g. an interrupted initial write).
            # Only our UUID directories are managed, and running jobs are pinned.
            paths={p.name:p for p in DATA.iterdir() if re.fullmatch('[a-f0-9]{32}',p.name) and p.is_dir() and not p.is_symlink()}
            for key in list(self.jobs):
                if key not in paths and key not in self.active:self.jobs.pop(key,None)
            completed=[]
            for key,path in paths.items():
                if key in self.active:continue
                job=self.jobs.get(key)
                # Never remove work that is still recorded as queued/running,
                # even if active bookkeeping is temporarily out of sync.
                if job and job.get('status') in ('queued','running'):continue
                finished=(job or {}).get('finished_at')
                if finished is None:
                    try:finished=(path/'job.json').stat().st_mtime
                    except OSError:finished=path.stat().st_mtime
                try:finished=float(finished)
                except (ValueError,TypeError):finished=path.stat().st_mtime
                completed.append((finished,key))
            completed.sort();remove=set()
            for finished,key in completed:
                if finished<time.time()-TTL_HOURS*3600:remove.add(key)
            for _,key in completed[:max(0,len(completed)-MAX_JOBS)]:remove.add(key)
            for key in remove:
                try:shutil.rmtree(paths[key])
                except FileNotFoundError:pass
                except OSError:
                    logger.exception('Could not remove expired job: %s',key);continue
                self.jobs.pop(key,None)

    def submit(self,payload,filename):
        if not self.capacity.acquire(blocking=False):raise HTTPException(429,'识别队列已满，请稍后重试。',headers={'Retry-After':'2'})
        try:
            with self.lock:
                identifier=uuid.uuid4().hex;directory=DATA/identifier;directory.mkdir()
                (directory/'uploaded-image').write_bytes(payload)
                job={'id':identifier,'status':'queued','stage':'等待识别','created_at':time.time(),'filename':filename[:200],
                     'progress':{'stage_code':'queued','percent':0,'current':None,'total':None}}
                self.jobs[identifier]=job;self.active.add(identifier)
                save_json(directory/'job.json',job)
                # Enqueue under the same lock as insertion: concurrent uploads
                # cannot reorder execution relative to displayed positions.
                self.pool.submit(self.work,identifier)
                return self.snapshot(identifier)
        except Exception:
            self.capacity.release();raise

    def work(self,identifier):
        directory=DATA/identifier
        try:
            with self.lock:self.jobs[identifier].update(status='running',stage='开始识别',
                    progress={'stage_code':'preparing','percent':None,'current':None,'total':None})
            def progress(update):
                with self.lock:
                    if isinstance(update,dict):
                        self.jobs[identifier]['stage']=update['stage']
                        self.jobs[identifier]['progress']=dict(update['progress'])
                    else:
                        self.jobs[identifier]['stage']=str(update)
            result=self.engine.run(directory/'uploaded-image',directory,identifier,progress)
            save_json(directory/'public-result.json',public_result(identifier,result))
            with self.lock:self.jobs[identifier].update(status='complete',stage='识别完成',
                    progress={'stage_code':'complete','percent':100,'current':None,'total':None})
        except Exception:
            logger.exception('Inference failed: %s',identifier)
            with self.lock:self.jobs[identifier].update(status='error',stage='处理失败',error='图片处理失败，请尝试清晰原图；详细原因见服务日志。',error_code='inference_failed',
                    progress={**self.jobs[identifier].get('progress',{}),'stage_code':'error'})
        finally:
            with self.lock:
                self.jobs[identifier]['finished_at']=time.time()
                save_json(directory/'job.json',self.jobs[identifier]);self.active.discard(identifier)
            self.capacity.release();self.cleanup()

    def snapshot(self,identifier):
        with self.lock:
            job=self.jobs.get(identifier)
            if job is None:raise HTTPException(404,'任务不存在或已过期。')
            result={**job,'progress':dict(job.get('progress',{}))}
            queue=self.queue_state();pending=queue.pop('pending_ids')
            position=pending.index(identifier)+1 if identifier in pending else 0 if job['status']=='running' else None
            result.update(queue_position=position,queue_ahead=queue['running']+position-1 if position else 0,
                          queue_total=queue['waiting'])
            if job['status']=='complete':result['result']=json.loads((DATA/identifier/'public-result.json').read_text())
            return result

    def queue_state(self):
        with self.lock:
            pending=[key for key,job in self.jobs.items() if key in self.active and job['status']=='queued']
            running=sum(key in self.active and job['status']=='running' for key,job in self.jobs.items())
            return {'pending_ids':pending,'running':running,'waiting':len(pending)}


@asynccontextmanager
async def lifespan(app):
    app.state.jobs=Jobs();app.state.upload_slots=asyncio.Semaphore(2)
    stop=asyncio.Event()
    async def maintain():
        while not stop.is_set():
            try:await asyncio.wait_for(stop.wait(),timeout=60)
            except asyncio.TimeoutError:
                try:await asyncio.to_thread(app.state.jobs.cleanup)
                except Exception:logger.exception('Scheduled result cleanup failed')
    maintenance=asyncio.create_task(maintain())
    try:yield
    finally:
        stop.set();await maintenance
        await asyncio.to_thread(app.state.jobs.pool.shutdown,wait=True,cancel_futures=False)


app=FastAPI(title='Flux Glyph API',version='1.0.0',lifespan=lifespan,docs_url=None,redoc_url=None,
            description='上传支付宝截图，检测文字区域，识别中文字体候选并下载标注图片。')


@app.exception_handler(HTTPException)
async def http_error(request,exception):
    return JSONResponse({'error':exception.detail},status_code=exception.status_code,headers=exception.headers)


@app.middleware('http')
async def protect(request,call_next):
    path=request.url.path
    protected=(path.startswith(('/api/','/assets/')) and path!='/api/health') or path=='/upload' or path.startswith('/upload/') or (path=='/' and request.method=='POST')
    if protected:
        supplied=request.headers.get('authorization','')
        cookie=request.cookies.get('flux_token','')
        if TOKEN and not (hmac.compare_digest(supplied.encode('utf-8'),('Bearer '+TOKEN).encode('utf-8')) or hmac.compare_digest(cookie.encode('utf-8'),TOKEN.encode('utf-8'))):
            return JSONResponse({'error':'需要访问令牌。'},status_code=401)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff';response.headers['Referrer-Policy']='no-referrer'
    response.headers['Cache-Control']='no-store' if protected else 'no-cache'
    response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
    return response


@app.get('/api/health')
def health():
    jobs=app.state.jobs
    queue=jobs.queue_state()
    return {'ok':True,'model_version':jobs.engine.version,'max_upload_mb':MAX_BYTES//1024//1024,
            'max_pixels':MAX_PIXELS,'workers':1,'queue_size':MAX_QUEUE,'authentication_required':bool(TOKEN),
            'running_jobs':queue['running'],'waiting_jobs':queue['waiting']}


@app.post('/auth')
async def authenticate(request:Request):
    body=bytearray()
    async for part in request.stream():
        if len(body)+len(part)>4096:raise HTTPException(413,'令牌请求过大。')
        body.extend(part)
    try:data=json.loads(body)
    except ValueError:raise HTTPException(400,'令牌请求需要 JSON。')
    if not isinstance(data,dict):raise HTTPException(400,'令牌请求需要 JSON 对象。')
    provided=data.get('token','')
    if not TOKEN or not isinstance(provided,str) or not hmac.compare_digest(provided.encode('utf-8'),TOKEN.encode('utf-8')):raise HTTPException(401,'令牌不正确。')
    response=JSONResponse({'ok':True});response.set_cookie('flux_token',TOKEN,httponly=True,samesite='strict',secure=request.url.scheme=='https',max_age=43200)
    return response


async def payload_from(request,filename=None):
    # Stream the entire request with a bounded allowance for multipart headers;
    # reject chunked requests that exceed it as well as announced large lengths.
    limit=MAX_BYTES+64*1024 if 'multipart/form-data' in request.headers.get('content-type','') else MAX_BYTES
    try:length=int(request.headers.get('content-length','0'))
    except ValueError:raise HTTPException(400,'错误的上传长度。')
    if length<0 or length>limit:raise HTTPException(413,'图片超过 8 MB 上传限制。')
    chunks=[];size=0
    async for part in request.stream():
        size+=len(part)
        if size>limit:raise HTTPException(413,'图片超过上传限制。')
        chunks.append(part)
    body=b''.join(chunks);request._body=body
    name=filename if filename is not None else unquote(request.headers.get('x-filename','image'))
    if 'multipart/form-data' in request.headers.get('content-type',''):
        try:
            async with request.form(max_files=1,max_fields=4,max_part_size=MAX_BYTES) as form:
                upload=form.get('file')
                if not isinstance(upload,UploadFile):raise HTTPException(400,'multipart 请求需要 file 字段。')
                name=upload.filename or 'image';body=await upload.read(MAX_BYTES+1)
        except MultipartParseError:raise HTTPException(400,'multipart 请求格式不正确。')
    try:await asyncio.to_thread(validate_image,body)
    except (ValueError,OSError,UnidentifiedImageError,Image.DecompressionBombError,Image.DecompressionBombWarning):raise HTTPException(400,'图片无法读取。支持 8 MB 内、1200 万像素内的 PNG/JPG/WebP 静态图。')
    return body,name


async def enqueue_job(request:Request,filename=None):
    if app.state.upload_slots.locked():raise HTTPException(429,'正在接收其他图片，请稍后重试。',headers={'Retry-After':'2'})
    async with app.state.upload_slots:
        payload,name=await payload_from(request,filename)
        return await asyncio.to_thread(app.state.jobs.submit,payload,name)


@app.post('/api/jobs',status_code=202)
async def create_job(request:Request):return await enqueue_job(request)


async def upload_compat(request:Request,wait:bool,filename=None):
    job=await enqueue_job(request,filename)
    if not wait:return JSONResponse(job,status_code=202)
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        state=app.state.jobs.snapshot(job['id'])
        if state['status']=='complete':return state['result']
        if state['status']=='error':return JSONResponse(state,status_code=422)
        await asyncio.sleep(.2)
    return JSONResponse(app.state.jobs.snapshot(job['id']),status_code=202)


@app.post('/api/predict')
async def predict(request:Request,wait:bool=True):return await upload_compat(request,wait)


@app.post('/')
async def upload_root(request:Request,wait:bool=True):return await upload_compat(request,wait)


@app.post('/upload')
async def upload_form(request:Request,wait:bool=True):return await upload_compat(request,wait)


@app.put('/upload/{filename:path}')
async def upload_binary(filename:str,request:Request,wait:bool=True):return await upload_compat(request,wait,filename)


@app.get('/api/jobs/{identifier}')
def get_job(identifier:str):return app.state.jobs.snapshot(identifier)


@app.get('/api/jobs/{identifier}/json')
def download_json(identifier:str):
    state=app.state.jobs.snapshot(identifier)
    if state['status']!='complete':raise HTTPException(409,'结果尚未完成。')
    return FileResponse(DATA/identifier/'public-result.json',media_type='application/json',filename=f'flux-glyph-{identifier}.json')


@app.get('/api/jobs/{identifier}/image')
def download_image(identifier:str):
    state=app.state.jobs.snapshot(identifier)
    if state['status']!='complete':raise HTTPException(409,'图片尚未完成。')
    return FileResponse(DATA/identifier/'annotated.png',media_type='image/png',filename=f'flux-glyph-{identifier}.png')


@app.get('/assets/{identifier}/{relative:path}')
def image_asset(identifier:str,relative:str):
    if not re.fullmatch('[a-f0-9]{32}',identifier):raise HTTPException(404,'图片不存在。')
    app.state.jobs.snapshot(identifier)
    root=(DATA/identifier).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix!='.png':raise HTTPException(404,'图片不存在。')
    return FileResponse(path,media_type='image/png')


@app.get('/')
def homepage():return FileResponse(ROOT/'web/index.html')


@app.get('/docs',include_in_schema=False)
def documentation():return FileResponse(ROOT/'web/docs.html')
app.mount('/static',StaticFiles(directory=ROOT/'web'),name='static')
