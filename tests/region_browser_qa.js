'use strict';
// UI contract fixtures only: this test never trains or evaluates font accuracy.
// Run: node tests/region_browser_qa.js (isolated localhost server + Chrome).
const fs = require('fs');
const path = require('path');
const http = require('http');
const {spawn} = require('child_process');
const root = path.resolve(__dirname, '..');
const out = path.join(root, 'artifacts/region-ui-qa');
const assert = (value, message) => { if (!value) throw Error(message); };
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const fixturePath = path.join(root, 'tests/fixtures/ui_title_billing_details.png');
const cropPng = fs.readFileSync(fixturePath);
const regions = Array.from({length:8}, (_, i) => {
  const box = [40, 50+i*105, 540, 125+i*105];
  return {id:'R'+String(i+1).padStart(3,'0'), quad:[[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]],
    source_bbox:box,detector_bbox:box,text:null,ocr_confidence:null,glyphs:[],crop_url:'/fixture/crop.png',
    font:{method:'region_neural_network',status:i===1?'uncertain':'candidate',family:i===1?null:['PingFang SC','SF Pro','MiSans'][i%3],
      scope:'Detected text region',reason_code:i===1?'below_score_gate':'region_neural_family_candidate',
      candidates:[{family:'PingFang SC',score:.9876},{family:'MiSans',score:.0101}]},
    text_style:{font_size_px_estimate:i===1?null:45.5,font_size_px_interval:[43,48],text_color_hex:i===1?null:'#112233',
      size:{status:i===1?'unavailable':'estimated'},color:{status:i===1?'unavailable':'estimated'}}};
});
const original = '<svg xmlns="http://www.w3.org/2000/svg" width="750" height="980"><rect width="750" height="980" fill="#f4f4f5"/>'+
  regions.map((region,i)=>'<image x="40" y="'+(50+i*105)+'" width="500" height="75" href="data:image/png;base64,'+cropPng.toString('base64')+'"/>').join('')+'</svg>';
const fixture = {id:'ui-fixture-only',width:750,height:980,image_url:'/fixture/original.svg',
  annotated_image_url:'/fixture/crop.png',summary:{detected_regions:regions.length},timing_seconds:{total:1.25},
  model_version:'ui-fixture-region-model',font_method:'region_neural_network',ocr_performed:false,regions};
let availability = 'available';
const groupedFonts = {families:['PingFang','SF Pro','Alipay Number','MiSans'],
  font_sources:{PingFang:['system'],'SF Pro':['system'],'Alipay Number':['asset'],MiSans:['asset']},
  font_label_groups:{PingFang:['PingFang SC','PingFang TC','PingFang HK']}};
let fontMetadata = groupedFonts;
const server = http.createServer((req, res) => {
  if (req.url === '/api/health') { res.setHeader('Content-Type','application/json'); res.end(JSON.stringify({model_version:'ui-fixture-region-model'})); return; }
  if (req.url === '/api/models/font') { res.setHeader('Content-Type','application/json');
    if(availability === 'locked') {res.statusCode=401;res.end('{}');return;}
    if(availability === 'failed') {res.statusCode=503;res.end('{}');return;}
    res.end(JSON.stringify({available:availability === 'available', version:'ui-fixture-v1', ...fontMetadata,
      input_shape:[null,1,64,256],download_url:'/api/models/font/download',bytes:1048576,ocr_required:false,
      usage:{install:'pip install -r requirements.txt',predict:'python predict.py text-region.png'}}));return; }
  if (req.url === '/api/models/font/download') {res.end('UI download route fixture');return;}
  if (req.url === '/fixture/original.svg') {res.setHeader('Content-Type','image/svg+xml');res.end(original);return;}
  if (req.url === '/fixture/crop.png') {res.setHeader('Content-Type','image/png');res.end(cropPng);return;}
  const filename = req.url === '/' ? path.join(root,'web/index.html') : req.url === '/docs' ? path.join(root,'web/docs.html') :
    req.url.startsWith('/static/') ? path.join(root,'web',req.url.slice(8).split('?')[0]) :
    null;
  if (!filename || !fs.existsSync(filename)) {res.statusCode=404;res.end('missing');return;}
  res.setHeader('Content-Type', filename.endsWith('.html')?'text/html; charset=utf-8':filename.endsWith('.js')?'text/javascript':filename.endsWith('.css')?'text/css':filename.endsWith('.svg')?'image/svg+xml':'image/png');
  fs.createReadStream(filename).pipe(res);
});
(async () => {
 fs.mkdirSync(out,{recursive:true});
 await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
 const port=server.address().port;const profile=fs.mkdtempSync(path.join(out,'chrome-'));
 const chrome=spawn(process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=0','--user-data-dir='+profile,'http://127.0.0.1:'+port],{stdio:['ignore','ignore','pipe']});
 let chromeErrors='';chrome.stderr.on('data',data=>chromeErrors+=String(data));let socket;
 try {
  const portFile=path.join(profile,'DevToolsActivePort');
  for(let i=0;i<120&&!fs.existsSync(portFile);i++) await sleep(100);
  assert(fs.existsSync(portFile),'Chrome did not launch: '+chromeErrors);
  const debugPort=fs.readFileSync(portFile,'utf8').split('\n')[0];
  let targets=[];for(let i=0;i<120&&!targets.some(x=>x.type==='page');i++) {targets=await (await fetch('http://127.0.0.1:'+debugPort+'/json/list')).json();await sleep(50);}
  socket=new WebSocket(targets.find(x=>x.type==='page').webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
  let sequence=0;const pending=new Map();const diagnostics=[];
  socket.onmessage=event=>{const value=JSON.parse(event.data);if(value.id){const task=pending.get(value.id);pending.delete(value.id);if(value.error)task.reject(Error(value.error.message));else task.resolve(value.result);}else if(value.method==='Runtime.exceptionThrown')diagnostics.push(value.params.exceptionDetails.text);};
  const command=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;const timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout '+method));},20000);pending.set(id,{resolve:value=>{clearTimeout(timer);resolve(value);},reject:error=>{clearTimeout(timer);reject(error);}});socket.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{const result=await command('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(result.exceptionDetails)throw Error(result.exceptionDetails.exception?.description||result.exceptionDetails.text);return result.result.value;};
  const wait=async expression=>{for(let i=0;i<120;i++){if(await evaluate(expression))return;await sleep(50);}throw Error('Timeout '+expression);};
  await command('Runtime.enable');await command('Page.enable');
  await command('Browser.grantPermissions',{origin:'http://127.0.0.1:'+port,permissions:['clipboardReadWrite','clipboardSanitizedWrite']});
  await wait('Boolean(window.FluxGlyphUI)');
  const reports=[];
  for(const [name,width,height] of [['desktop',1280,900],['mobile',390,844],['small-mobile',320,700]]) {
    await command('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
    await evaluate('window.FluxGlyphUI.show('+JSON.stringify(fixture)+')');
    await wait("document.querySelectorAll('.region-thumbnail img').length===8 && [...document.querySelectorAll('.region-thumbnail img')].every(i=>i.complete&&i.naturalWidth>0) && document.getElementById('download-model').hasAttribute('href')");
    let result=await evaluate(`(() => ({ocr:document.querySelectorAll('#ocr-output,#copy-ocr,.ocr-panel,.ocr-confidence').length,titles:[...document.querySelectorAll('.region-heading b')].map(x=>x.textContent),thumbs:document.querySelectorAll('.region-thumbnail img').length,overflow:document.documentElement.scrollWidth>innerWidth,modelBeforeUpload:document.getElementById('font-model').getBoundingClientRect().bottom<=document.querySelector('.upload-help').getBoundingClientRect().top,usageCollapsed:!document.getElementById('model-usage').open,download:document.getElementById('download-model').getAttribute('href'),usage:document.getElementById('model-usage-command').textContent,json:document.getElementById('json-output').textContent}))()`);
    assert(result.ocr===0,'OCR UI remained');assert(result.titles.join()===regions.map(x=>x.id).join(),'Region IDs missing');assert(!result.overflow,'Horizontal overflow '+name);assert(result.modelBeforeUpload,'Model download is not above upload '+name);assert(result.usageCollapsed,'Usage should initially be compact '+name);assert(result.download==='/api/models/font/download','Missing same-origin model link');assert(result.usage.includes('python predict.py text-region.png'),'Missing model usage');assert(JSON.parse(result.json).ocr_performed===false,'Original no-OCR JSON changed');
    result.fontSources=await evaluate(`(() => ({system:document.querySelector('[data-font-source="system"]').textContent,asset:document.querySelector('[data-font-source="asset"]').textContent,note:document.getElementById('model-label-note').textContent,noteVisible:!document.getElementById('model-label-note').hidden}))()`);
    assert(result.fontSources.system==='系统内置字体：PingFang · SF Pro','System fonts must use recorded sources');
    assert(result.fontSources.asset==='应用自带字体：Alipay Number · MiSans','Alipay Number must be shown as an app font');
    assert(result.fontSources.noteVisible&&result.fontSources.note.includes('覆盖简体／繁体，未细分地区版本'),'Grouped PingFang scope missing');
    await evaluate("document.querySelector('#regions .region').click()");
    result.detail=await evaluate(`(() => ({title:document.querySelector('.detail-preview h2').textContent,glyphs:document.querySelectorAll('.glyph,.detail-glyphs').length,score:document.querySelector('.dist').textContent,style:document.querySelector('.text-style-detail').textContent,visible:!document.getElementById('detail-panel').hidden,scope:document.getElementById('detail').textContent.includes('整体外观')}))()`);
    assert(result.detail.title===regions[0].id&&result.detail.glyphs===0&&result.detail.visible,'Detail region-only failed');assert(result.detail.score.includes('0.9876'),'Region CNN score not shown');assert(result.detail.style.includes('45.5')&&result.detail.style.includes('#112233'),'Style not shown');assert(result.detail.scope,'Region scope missing');
    await evaluate("window.FluxGlyphUI.setLanguage('en')");
    assert(await evaluate("document.getElementById('font-model-title').textContent==='Font model' && document.querySelector('.dist').textContent.includes('model scores')"),'English model UI missing');
    assert(await evaluate(`document.querySelector('[data-font-source="system"]').textContent==='Built-in system fonts: PingFang · SF Pro' && document.querySelector('[data-font-source="asset"]').textContent==='App-bundled fonts: Alipay Number · MiSans' && document.getElementById('model-label-note').textContent.includes('regional variants are not classified separately')`),'English font source categories or PingFang scope missing');
    assert(await evaluate('document.getElementById("json-output").textContent === '+JSON.stringify(result.json)),'Language changed raw JSON');
    await evaluate("document.getElementById('copy-json').click()");
    await wait("document.getElementById('copy-status').textContent.includes('copied')");
    assert(await evaluate('navigator.clipboard.readText()')===result.json,'JSON clipboard changed raw result');
    await evaluate("window.FluxGlyphUI.setLanguage('zh');document.getElementById('result-section').scrollIntoView()");
    const screen=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(out,name+'.png'),Buffer.from(screen.data,'base64'));
    await evaluate("window.scrollTo(0,0)");
    const modelScreen=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(path.join(out,name+'-model.png'),Buffer.from(modelScreen.data,'base64'));
    await evaluate("document.getElementById('close-detail').click()");assert(await evaluate("document.getElementById('detail-panel').hidden"),'Cannot close detail');reports.push({name,width,passed:true,...result});
  }
  const reasonCases = {below_score_gate:'分数未达到确认门槛',ambiguous_neural_families:'前两名字体',mixed_or_ambiguous_region:'不同图像片段',region_too_long:'文字区域过长',low_quality_region:'图像质量不足',nonuniform_region_background:'背景颜色不均匀'};
  for(const [code,phrase] of Object.entries(reasonCases)) {
    const value={...fixture,regions:[{...regions[0],font:{...regions[0].font,status:'uncertain',family:null,reason_code:code}}]};
    await evaluate('window.FluxGlyphUI.setLanguage("zh");window.FluxGlyphUI.show('+JSON.stringify(value)+');document.querySelector("#regions .region").click()');
    const explanation=await evaluate('document.getElementById("font-reason").textContent');
    assert(explanation.includes(phrase)&&!explanation.includes('切分'),'Missing specific region uncertainty reason '+code);
  }
  const availabilityChecks=[];
  for(const mode of ['unavailable','failed','locked']) {
    availability=mode;
    await evaluate("document.getElementById('refresh-model').click()");
    await wait("document.getElementById('font-model').getAttribute('aria-busy')==='false'");
    const state=await evaluate(`(() => ({disabled:document.getElementById('download-model').getAttribute('aria-disabled'),href:document.getElementById('download-model').getAttribute('href'),reason:document.getElementById('model-info').textContent,retryEnabled:!document.getElementById('refresh-model').disabled,authVisible:!document.getElementById('auth-panel').hidden}))()`);
    assert(state.disabled==='true'&&!state.href&&state.retryEnabled,'Unavailable model must retain disabled download and allow retry: '+mode);
    assert(state.reason.includes(mode==='locked'?'访问令牌':mode==='failed'?'刷新状态':'暂无可下载'),'Availability reason must explain the failure: '+mode);
    if(mode==='locked')assert(state.authVisible,'Locked download must expose the existing unlock panel');
    availability='available';
    await evaluate("document.getElementById('refresh-model').click()");
    await wait("document.getElementById('font-model').getAttribute('aria-busy')==='false' && document.getElementById('download-model').getAttribute('aria-disabled')==='false'");
    assert(await evaluate("document.getElementById('download-model').getAttribute('href')==='/api/models/font/download'"),'Refresh did not recover download without reloading the page');
    availabilityChecks.push({mode,...state,recovered:true});
  }
  await evaluate("document.querySelector('#model-usage summary').click()");
  assert(await evaluate("document.getElementById('model-usage').open && document.getElementById('model-usage-command').getBoundingClientRect().height>0"),'Expandable local usage did not open');
  await evaluate("window.scrollTo(0,0)");
  const sourceScreen=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  fs.writeFileSync(path.join(out,'small-mobile-model-sources.png'),Buffer.from(sourceScreen.data,'base64'));
  assert(await evaluate('document.documentElement.scrollWidth<=innerWidth'),'Expanded font source list overflows mobile viewport');
  const sourceCompatibilityChecks=[];
  for(const mode of ['legacy-omitted','legacy-empty','partial']) {
    fontMetadata=mode==='partial'
      ? {families:['PingFang','SF Pro','Alipay Number'],font_sources:{PingFang:['system'],'SF Pro':['asset'],'Not in model':['asset']}}
      : {families:['PingFang SC','SF Pro','Alipay Number'],...(mode==='legacy-empty'?{font_sources:{},font_label_groups:{}}:{})};
    await evaluate("document.getElementById('refresh-model').click()");
    await wait("document.getElementById('font-model').getAttribute('aria-busy')==='false'");
    const state=await evaluate(`(() => ({rows:[...document.querySelectorAll('#model-families p')].map(p=>({source:p.dataset.fontSource||null,text:p.textContent})),noteHidden:document.getElementById('model-label-note').hidden,download:document.getElementById('download-model').getAttribute('href')}))()`);
    assert(state.noteHidden,'Missing grouped label metadata must hide PingFang note: '+mode);
    assert(state.download==='/api/models/font/download','Optional source fields must not break download: '+mode);
    if(mode==='partial') {
      assert(JSON.stringify(state.rows)===JSON.stringify([{source:'system',text:'系统内置字体：PingFang'},{source:'asset',text:'应用自带字体：SF Pro'},{source:null,text:'其他可识别字体：Alipay Number'}]),'Partial metadata must not guess font sources from familiar names or include unlisted families');
    } else {
      assert(JSON.stringify(state.rows)===JSON.stringify([{source:null,text:'可识别字体：PingFang SC · SF Pro · Alipay Number'}]),'Old model metadata must retain the unclassified font list');
    }
    sourceCompatibilityChecks.push({mode,...state});
  }

  assert(!diagnostics.length,'JS exceptions '+diagnostics.join());
  fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({kind:'mocked region-result browser UI only, not model accuracy',passed:true,reports,availabilityChecks,sourceCompatibilityChecks,diagnostics},null,2));
  console.log(JSON.stringify({passed:true,viewports:reports.map(x=>x.name),output:out}));
 } finally {if(socket)socket.close();chrome.kill('SIGTERM');await new Promise(resolve=>chrome.exitCode!==null?resolve():chrome.once('exit',resolve));server.close();fs.rmSync(profile,{recursive:true,force:true});}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});
