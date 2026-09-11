"""Sequential real-image CPU/resource benchmark; no font accuracy ground truth."""
import argparse,hashlib,io,json,subprocess,time
from collections import Counter
from pathlib import Path
import httpx
import numpy as np
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--url',default='http://127.0.0.1:9000');p.add_argument('--container',default='flux-glyph-api-1');p.add_argument('--output',type=Path,default=Path('docs/resource-validation.json'));args=p.parse_args()
sources=[json.loads(x) for x in args.sources.read_text().splitlines() if x.strip()]
client=httpx.Client(base_url=args.url,timeout=150)
health=client.get('/api/health').json();records=[];totals=Counter();families=Counter()
archive=Path('artifacts/raw-benchmark');archive.mkdir(parents=True,exist_ok=True)
for i,row in enumerate(sources):
 body=Path(row.get('original_copy') or row['image']).read_bytes()
 assert hashlib.sha256(body).hexdigest()==row['source_sha256']
 begin=time.perf_counter();response=client.post('/api/predict?wait=true',content=body,headers={'Content-Type':'application/octet-stream'});response.raise_for_status()
 assert response.status_code==200,response.text
 result=response.json();assert result['source_sha256']==row['source_sha256']
 elapsed=time.perf_counter()-begin
 totals.update(result['summary']);families.update(r['font']['family'] for r in result['regions'] if r['font']['family'])
 record={'source_id':row['id'],'source_sha256':row['source_sha256'],'domain':row.get('source_domain'),'job_id':result['id'],
         'size':[result['width'],result['height']],'summary':result['summary'],'timing_seconds':result['timing_seconds'],'http_seconds':elapsed}
 records.append(record)
 (archive/(row['id']+'.json')).write_text(json.dumps(result,ensure_ascii=False)+'\n')
 if (i+1)%10==0:print(f'{i+1}/{len(sources)} completed',flush=True)

# A separate maximum-pixel input checks allocation limits, not font accuracy.
with Image.open(sources[0].get('original_copy') or sources[0]['image']) as image:
 stress=image.convert('RGB').resize((2000,6000),Image.Resampling.BICUBIC)
 buffer=io.BytesIO();stress.save(buffer,'PNG');assert len(buffer.getvalue())<8*1024*1024
response=client.post('/api/predict?wait=true',content=buffer.getvalue(),headers={'Content-Type':'image/png'});response.raise_for_status();assert response.status_code==200
limit_case=response.json()
info=json.loads(subprocess.check_output(['docker','inspect',args.container],text=True))[0]
def cgroup(name):
 return int(subprocess.check_output(['docker','exec',args.container,'cat','/sys/fs/cgroup/'+name],text=True))
native_arch=subprocess.check_output(['docker','exec',args.container,'uname','-m'],text=True).strip()
report={'status':'passed','claim_scope':'local Docker Linux resource and coverage test, not true-font accuracy or cloud-host speed',
 'health':health,'container':{'architecture':native_arch,'cpu_limit':info['HostConfig']['NanoCpus']/1e9,
 'memory_limit_bytes':info['HostConfig']['Memory'],'memory_peak_bytes':cgroup('memory.peak'),'memory_current_bytes':cgroup('memory.current'),
 'oom_killed':info['State']['OOMKilled'],'restart_count':info['RestartCount'],'running':info['State']['Running'],
 'readonly_rootfs':info['HostConfig']['ReadonlyRootfs'],'image_id':info['Image']},
 'count':len(records),'domains':dict(Counter(x['domain'] for x in records)),
 'latency_seconds':{'pipeline_median':float(np.median([r['timing_seconds']['total'] for r in records])),
 'pipeline_p95':float(np.percentile([r['timing_seconds']['total'] for r in records],95)),
 'http_median':float(np.median([r['http_seconds'] for r in records])),
 'http_p95':float(np.percentile([r['http_seconds'] for r in records],95))},
 'coverage_counts':dict(totals),'emitted_font_counts':dict(families),
 'maximum_pixel_probe':{'size':[limit_case['width'],limit_case['height']],'summary':limit_case['summary'],'timing_seconds':limit_case['timing_seconds']},'records':records}
assert report['container']['cpu_limit']==2 and report['container']['memory_limit_bytes']==1536*1024*1024
assert not report['container']['oom_killed'] and report['container']['restart_count']==0
args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))
