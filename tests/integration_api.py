"""Real HTTP tests against the container (run with --url, optional --sources JSONL)."""
import argparse,hashlib,io,json,time
from pathlib import Path
import httpx
import numpy as np
from PIL import Image,ImageOps

p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:9000');p.add_argument('--sources',type=Path);p.add_argument('--output',type=Path,default=Path('docs/api-validation.json'));args=p.parse_args()
root=args.url.rstrip('/');client=httpx.Client(base_url=root,timeout=140);health=client.get('/api/health');health.raise_for_status()
fixtures=[]
if args.sources:
 for row in [json.loads(x) for x in args.sources.read_text().splitlines() if x.strip()]:fixtures.append((row['id'],Path(row['image']).read_bytes()))
else:fixtures=[('ui-title',Path('tests/fixtures/ui_title_billing_details.png').read_bytes())]
with Image.open('tests/fixtures/ui_title_billing_details.png') as title:
 exif=Image.Exif();exif[274]=6
 encoded=io.BytesIO();title.transpose(Image.Transpose.ROTATE_90).save(encoded,'JPEG',quality=95,exif=exif)
 fixtures.append(('exif-rotated-title',encoded.getvalue()))
b=io.BytesIO();Image.new('RGB',(400,600),'white').save(b,'PNG');fixtures.append(('blank',b.getvalue()))
records=[]
for i,(name,body) in enumerate(fixtures):
 if i%2==0:
  response=client.post('/api/predict?wait=true',files={'file':('test.png',body,'image/png')})
  assert response.status_code==200,response.text
  result=response.json();identifier=result['id']
 else:
  response=client.post('/api/jobs',content=body,headers={'Content-Type':'application/octet-stream','X-Filename':'..%2F..%2Finjected.png'})
  assert response.status_code==202,response.text
  identifier=response.json()['id'];deadline=time.monotonic()+120
  while True:
   job=client.get('/api/jobs/'+identifier).json()
   if job['status']=='complete':result=job['result'];break
   assert job['status']!='error',job
   assert time.monotonic()<deadline
   time.sleep(.2)
 with Image.open(io.BytesIO(body)) as source:original=ImageOps.exif_transpose(source).convert('RGB')
 assert result['source_sha256']==hashlib.sha256(body).hexdigest()
 shown=client.get(result['image_url']);shown.raise_for_status()
 with Image.open(io.BytesIO(shown.content)) as image:assert np.array_equal(np.asarray(image),np.asarray(original))
 png=client.get(result['annotated_image_url']);png.raise_for_status();assert 'attachment' in png.headers.get('content-disposition','')
 with Image.open(io.BytesIO(png.content)) as image:
  assert image.size==original.size
  if result['regions']:assert not np.array_equal(np.asarray(image),np.asarray(original))
 download=client.get(result['download_json_url']);assert download.status_code==200 and download.json()==result
 assert result['summary']['detected_regions']==len(result['regions'])
 for r in result['regions']:
  assert r['font']['status'] in ('supported','candidate','uncertain','out_of_scope')
  c=client.get(r['crop_url']);assert c.status_code==200
  with Image.open(io.BytesIO(c.content)) as crop:assert np.array_equal(np.asarray(crop),np.asarray(original.crop(r['source_bbox'])))
  for g in r['glyphs']:
   if not g.get('crop_url'):continue
   glyph=client.get(g['crop_url']);glyph.raise_for_status()
   expected=original.crop(g['source_bbox'])
   if g['source_rotation_degrees']==180:expected=expected.transpose(Image.Transpose.ROTATE_180)
   with Image.open(io.BytesIO(glyph.content)) as actual:assert np.array_equal(np.asarray(actual),np.asarray(expected))
 if name=='blank':assert not result['regions']
 records.append({'name':name,'id':identifier,'summary':result['summary'],'timing_seconds':result['timing_seconds'],'size':[result['width'],result['height']]})
 print(name,result['summary'],flush=True)
assert client.post('/api/jobs',content=b'badimage').status_code==400
assert client.get('/assets/'+'f'*32+'/../../models/MANIFEST.json').status_code==404
assert client.get('/api/jobs/missing').status_code==404
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps({'status':'passed','health':health.json(),'records':records,'checks':['multipart synchronous predict','raw binary asynchronous job','raw byte SHA','EXIF orientation','original and ROI exact pixels','every extracted glyph exact source pixels','same-size annotated PNG changed only on copy','JSON download exact result','blank image','invalid image rejection','missing/traversal path rejection']},ensure_ascii=False,indent=2)+'\n')
