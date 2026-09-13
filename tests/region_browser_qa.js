'use strict';
// UI contract fixtures only: this test never trains or evaluates font accuracy.
// Run: node tests/region_browser_qa.js (isolated localhost server + Chrome).
const fs = require('fs');
const path = require('path');
const http = require('http');
const {spawn} = require('child_process');
const root = path.resolve(__dirname, '..');
const out = path.resolve(process.env.FLUX_QA_OUTPUT || path.join(root, 'artifacts/region-ui-qa'));
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
// Deliberately unsorted and contradictory candidates check presentation only.
// Neither the selected display name nor its score may rewrite the server result.
const scoreUiCases = [
  {id:'score-low',status:'uncertain',family:null,reason_code:'below_score_gate',
    candidates:[{family:'MiSans',score:.3441},{family:'',score:.99},{family:'PingFang',score:.6559}],
    expected:[['PingFang','0.6559'],['MiSans','0.3441']]},
  {id:'score-patch-conflict',status:'uncertain',family:null,reason_code:'mixed_or_ambiguous_region',
    candidates:[{family:'SF Pro',score:.89},{family:'MiSans',score:.11}],expected:[['SF Pro','0.8900'],['MiSans','0.1100']]},
  {id:'score-zero',method:'neural_network',status:'uncertain',family:null,reason_code:'below_score_gate',score:.9999,
    candidates:[{family:'SF Pro',score:0}],expected:[['SF Pro','0.0000']]},
  {id:'score-invalid',status:'uncertain',family:'Stale family',reason_code:'invalid_neural_output',score:1,distance:.0001,
    candidates:[{family:'NaN score',score:NaN},{family:'Infinite score',score:Infinity},{family:'Null score',score:null},
      {family:'String score',score:'0.9999'},{family:'Negative score',score:-.01},{family:'Excessive score',score:1.01},
      {family:'Missing score'},{family:'',score:1},{family:'   ',score:1},{family:42,score:1},{family:null,score:1},null],expected:[]},
  {id:'score-family-conflict',status:'candidate',family:'MiSans',reason_code:'region_neural_family_candidate',
    candidates:[{family:'MiSans',score:.07},{family:'PingFang',score:.92},{family:'SF Pro',score:.01}],
    expected:[['PingFang','0.9200'],['MiSans','0.0700'],['SF Pro','0.0100']]},
  {id:'score-tie',method:'neural_network',status:'uncertain',family:null,reason_code:'ambiguous_neural_families',
    candidates:[{family:'Fourth',score:.05},{family:'First tie',score:.5},{family:'Third',score:.1},{family:'Second tie',score:.5}],
    expected:[['First tie','0.5000'],['Second tie','0.5000'],['Third','0.1000']]},
  {id:'score-legacy',method:'reference_matching',status:'supported',family:'Legacy Reference',reason_code:'pingfang_supported',
    candidates:[{family:'Legacy Reference',distance:.0066,score:.01},{family:'Other reference',distance:.0162,score:.99}],
    expected:[['Legacy Reference','0.0066'],['Other reference','0.0162']]},
  {id:'score-absent',status:'uncertain',family:null,reason_code:'no_samples',score:.88,distance:.0001,candidates:[],expected:[]}
];
const scoreFixture = {...fixture,id:'ui-score-fixture-only',regions:scoreUiCases.map(({id,expected,...font},i)=>({
  ...regions[i],id,font:{method:'region_neural_network',scope:'Detected text region',...font}
}))};
const residualCandidates = [{family:'PingFang',score:.9998},{family:'SF Pro',score:.0002}];
const rejectionUiCases = [
  {id:'unknown-clean',kind:'unknown',status:'out_of_scope',family:null,score:null,candidates:[],
    reason_code:'unknown_font_rejected',rejection:{method:'neural_network',status:'rejected',known_score:.12,min_known_score:.8}},
  {id:'unknown-residual',kind:'unknown',status:'out_of_scope',family:'PingFang',score:.9998,candidates:residualCandidates,
    components:[{family:'PingFang',status:'candidate',scope:'Latin letters and digits only'}],
    reason_code:'unknown_font_rejected',rejection:{method:'neural_network',status:'rejected',known_score:.12,min_known_score:.8}},
  {id:'unknown-status-only',kind:'unknown',status:'candidate',family:'PingFang',score:.9998,candidates:residualCandidates,
    reason_code:'below_score_gate',rejection:{method:'neural_network',status:'rejected',known_score:.12,min_known_score:.8}},
  {id:'invalid-status-only',kind:'invalid',status:'uncertain',family:'PingFang',score:.9998,candidates:residualCandidates,
    reason_code:'below_score_gate',rejection:{method:'neural_network',status:'unavailable',known_score:null,min_known_score:.8}},
  {id:'invalid-reason-priority',kind:'invalid',status:'uncertain',family:'PingFang',score:.9998,candidates:residualCandidates,
    reason_code:'invalid_rejection_output',rejection:{method:'neural_network',status:'rejected',known_score:null,min_known_score:.8}},
  {id:'unknown-reason-priority',kind:'unknown',status:'out_of_scope',family:'PingFang',score:.9998,candidates:residualCandidates,
    reason_code:'unknown_font_rejected',rejection:{method:'neural_network',status:'passed',known_score:1,min_known_score:.8}},
  {id:'known-passed',kind:'known',status:'candidate',family:'PingFang',score:.9,
    candidates:[{family:'PingFang',score:.9},{family:'SF Pro',score:.07},{family:'MiSans',score:.03}],
    reason_code:'region_neural_family_candidate',rejection:{method:'neural_network',status:'passed',known_score:.99,min_known_score:.8}},
  {id:'known-legacy-v1',kind:'known',status:'candidate',family:'SF Pro',score:.87,
    candidates:[{family:'SF Pro',score:.87},{family:'PingFang',score:.1},{family:'MiSans',score:.03}],reason_code:'region_neural_family_candidate'}
];
const rejectionFixture = {...fixture,id:'ui-rejection-fixture-only',regions:rejectionUiCases.map(({id,kind,...font},i)=>({
  ...regions[i],id,font:{method:'region_neural_network',scope:'Detected text region',font_size_px_estimate:i===0?null:42,...font},
  text_style:{font_size_px_estimate:i===0?null:42,font_size_px_interval:i===0?null:[40,44],text_color_hex:'#224466',
    size:{status:i===0?'unavailable':'estimated'},color:{status:'estimated'}}
}))};
let availability = 'available';
const groupedFonts = {families:['PingFang','SF Pro','Alipay Number','MiSans'],
  font_sources:{PingFang:['system'],'SF Pro':['system'],'Alipay Number':['asset'],MiSans:['asset']},
  font_label_groups:{PingFang:['PingFang SC','PingFang TC','PingFang HK']}};
let fontMetadata = groupedFonts;
let androidAvailability = true;
let androidDelay = 0;
let androidJobComplete = false;
const previewMetadata = {release_tier:'experimental',stable_validation_passed:false,test_passed:false,
  validation:{named_precision:.9259954921111946,known_correct_coverage:.6847222222222222,
    unknown_wrongly_named:174,unknown_views:1200,unknown_test_families:['Smiley Sans']}};
let androidRelease = previewMetadata;
const modeUploads = [];
const androidFixture = {...fixture,model_release_tier:'experimental',stable_validation_passed:false,id:'android-ui-job',font_mode:'android',model_version:'android-ui-fixture',
  device_inference_performed:false,regions:[{...regions[0],font:{method:'region_neural_network',font_mode:'android',
    status:'candidate',family:'Noto Sans CJK SC',score:.8765,candidates:[{family:'Noto Sans CJK SC',score:.8765}],
    reason_code:'region_neural_family_candidate'}}]};
const server = http.createServer((req, res) => {
  if (req.url === '/api/models/font?mode=android') {
    const body={...androidRelease,available:androidAvailability,font_mode:'android',version:'android-ui-fixture',
      families:['Noto Sans CJK SC','Noto Serif CJK SC','LXGW WenKai','WenQuanYi Micro Hei','ZCOOL KuaiLe','ZCOOL XiaoWei','ZCOOL QingKe HuangYou','Ma Shan Zheng','Roboto'],font_sources:{'Roboto':['system'],'LXGW WenKai':['asset']},
      font_label_groups:{'Noto Sans CJK SC':['Noto Sans CJK SC','Source Han Sans']},
      download_url:'/api/models/font/download?mode=android',bytes:1048576,
      usage:{install:'pip install -r requirements.txt',predict:'python predict.py text-region.png'}};
    setTimeout(()=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(body));},androidDelay);return;
  }
  if (req.url === '/api/models/font/download?mode=android') {res.end('Independent Android UI download fixture');return;}
  if (req.url === '/api/jobs?mode=android' && req.method === 'POST') {
    let bytes=0;req.on('data',part=>bytes+=part.length);req.on('end',()=>{
      modeUploads.push({mode:'android',bytes});res.statusCode=202;res.setHeader('Content-Type','application/json');
      res.end(JSON.stringify({id:'android-ui-job',font_mode:'android',status:'queued',progress:{stage_code:'queued',percent:0}}));
    });return;
  }
  if (req.url === '/api/jobs/android-ui-job') {
    res.setHeader('Content-Type','application/json');res.end(JSON.stringify({id:'android-ui-job',font_mode:'android',
      status:androidJobComplete?'complete':'queued',progress:{stage_code:androidJobComplete?'complete':'queued',percent:androidJobComplete?100:0},
      ...(androidJobComplete?{result:androidFixture}:{})}));return;
  }
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
  for(const code of ['low_quality_region','nonuniform_region_background','region_too_long','too_many_regions']) {
    const value={...fixture,regions:[{...regions[0],font:{...regions[0].font,status:'uncertain',family:null,reason_code:code,
      rejection:{method:'neural_network',status:'unavailable',known_score:null,min_known_score:.8}}}]};
    await evaluate('window.FluxGlyphUI.show('+JSON.stringify(value)+');document.querySelector("#regions .region").click()');
    const state=await evaluate('({reason:document.getElementById("font-reason").textContent,label:document.getElementById("font-label").textContent,scores:document.querySelectorAll(".font-score,.dist").length,size:document.querySelectorAll(".text-size").length})');
    assert(state.reason.includes(code==='too_many_regions'?'文字区域过多':reasonCases[code]),'Preprocessing reason must survive an unavailable rejection check: '+code);
    assert(state.label==='未生成评分'&&state.scores===0&&state.size===0,'Unavailable preprocessing must not leak candidate names, scores or size: '+code);
  }
  const scoreChecks=[];
  const statusLabels={zh:{uncertain:'待确认',candidate:'候选',supported:'支持'},en:{uncertain:'Review',candidate:'Candidate',supported:'Supported'}};
  const statusColors={uncertain:'rgb(173, 98, 0)',candidate:'rgb(35, 90, 180)',supported:'rgb(20, 128, 74)'};
  const scoreReasons={
    below_score_gate:{zh:'模型分数未达到确认门槛',en:'model score is below the gate'},
    mixed_or_ambiguous_region:{zh:'不同图像片段的字体判断不一致',en:'Different image patches within the region disagree'},
    invalid_neural_output:{zh:'输出异常',en:'invalid output'},
    region_neural_family_candidate:{zh:'达到当前门槛',en:'passes the current score and separation gates'},
    ambiguous_neural_families:{zh:'模型评分过于接近',en:'similar model scores'},
    pingfang_supported:{zh:'逐字候选一致',en:'agree across character candidates'},
    no_samples:{zh:'没有可用于字体判断',en:'No reliable glyphs'}
  };
  await command('Emulation.setDeviceMetricsOverride',{width:320,height:700,deviceScaleFactor:1,mobile:false});
  // JSON cannot encode NaN or Infinity. Restore these two deliberate in-memory
  // malformed-result cases before showing the same result object to the UI.
  await evaluate(`(() => {
    window.__scoreUiResult=${JSON.stringify(scoreFixture)};
    window.__scoreUiResult.regions[3].font.candidates[0].score=NaN;
    window.__scoreUiResult.regions[3].font.candidates[1].score=Infinity;
    window.__scoreUiJson=JSON.stringify(window.__scoreUiResult,null,2);
    window.FluxGlyphUI.show(window.__scoreUiResult);
  })()`);
  for(const language of ['zh','en']) {
    await evaluate(`window.FluxGlyphUI.setLanguage(${JSON.stringify(language)});document.getElementById('close-detail').click();document.getElementById('regions').scrollIntoView({block:'start'})`);
    const state=await evaluate(`(() => ({
      rows:[...document.querySelectorAll('#regions .region')].map(row=>({
        id:row.dataset.regionId,prediction:row.querySelector('.font-prediction')?.textContent??null,
        label:row.querySelector('.region-description > small')?.textContent??null,
        score:row.querySelector('.font-score')?.textContent??null,
        status:row.querySelector('.tag').textContent,statusClass:row.querySelector('.tag').className,
        color:getComputedStyle(row.querySelector('.tag')).color,classes:row.className,
        overlay:document.querySelector('#overlay [data-region-id="'+row.dataset.regionId+'"]').getAttribute('class')
      })),accepted:document.querySelector('#summary .summary-chip.supported').textContent,
      overflow:document.documentElement.scrollWidth>innerWidth
    }))()`);
    assert(!state.overflow,'Score list overflows 320px in '+language);
    assert(state.accepted===(language==='zh'?'已识别 2':'Identified 2'),'Displaying neural predictions must not inflate accepted count: '+language);
    const listScreen=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    fs.writeFileSync(path.join(out,'score-small-mobile-'+language+'-list.png'),Buffer.from(listScreen.data,'base64'));
    for(const [index,testCase] of scoreUiCases.entries()) {
      const row=state.rows[index];const neural=testCase.method!=='reference_matching';
      const prediction=neural&&testCase.expected.length?testCase.expected[0]:null;
      const expectedLabel=prediction?(language==='zh'?'最接近：':'Closest match: ')+prediction[0]:neural?(language==='zh'?'未生成评分':'No model score'):testCase.family;
      const expectedScore=prediction?(language==='zh'?'模型评分 ':'Model score ')+prediction[1]:null;
      const context=testCase.id+' / '+language;
      assert(row.id===testCase.id&&row.label===expectedLabel,'Incorrect list prediction '+context+': '+row.label);
      assert(row.prediction===(neural?expectedLabel:null),'Legacy labels must not be neural predictions '+context);
      assert(row.score===expectedScore,'Incorrect list model score '+context+': '+row.score);
      assert(row.status===statusLabels[language][testCase.status]&&row.statusClass==='tag '+testCase.status,'List status changed '+context);
      assert(row.color===statusColors[testCase.status]&&row.classes.split(' ').includes(testCase.status)&&row.overlay.split(' ').includes(testCase.status),'Original status color/class changed '+context);
      await evaluate('document.querySelector('+JSON.stringify('#regions [data-region-id="'+testCase.id+'"]')+').click()');
      const detail=await evaluate(`(() => ({label:document.getElementById('font-label').textContent,
        score:document.getElementById('font-score')?.textContent??null,
        status:document.querySelector('.detail-verdict .tag').textContent,
        statusClass:document.querySelector('.detail-verdict .tag').className,
        color:getComputedStyle(document.querySelector('.detail-verdict .tag')).color,
        reason:document.getElementById('font-reason').textContent,
        topHeading:document.querySelector('.dist > p')?.textContent??null,
        top:[...document.querySelectorAll('.dist > span')].map(span=>span.textContent),
        overflow:document.documentElement.scrollWidth>innerWidth,
        rawUnchanged:document.getElementById('json-output').textContent===window.__scoreUiJson&&JSON.stringify(window.__scoreUiResult,null,2)===window.__scoreUiJson,
        nonfiniteUnchanged:Number.isNaN(window.__scoreUiResult.regions[3].font.candidates[0].score)&&window.__scoreUiResult.regions[3].font.candidates[1].score===Infinity
      }))()`);
      assert(detail.label===expectedLabel&&detail.score===expectedScore,'List/detail prediction or score differ '+context);
      assert(detail.status===row.status&&detail.statusClass===row.statusClass&&detail.color===row.color,'Detail must retain the real verdict, including absent-family cases '+context);
      assert(detail.reason.includes(scoreReasons[testCase.reason_code][language]),'Original reason changed '+context);
      assert(JSON.stringify(detail.top)===JSON.stringify(testCase.expected.flat()),'Top 3 must use valid stable score order, or unchanged legacy distances '+context+': '+detail.top.join());
      if(testCase.expected.length)assert(detail.topHeading.includes(neural?(language==='zh'?'模型分数':'model scores'):(language==='zh'?'原始距离':'raw distances')),'Wrong Top 3 quantity label '+context);
      else assert(detail.topHeading===null,'Invalid scores must not produce a Top 3 table '+context);
      assert(!detail.overflow,'Score detail overflows 320px '+context);
      assert(detail.rawUnchanged&&detail.nonfiniteUnchanged,'Rendering/sorting/language changed original result '+context);
      if(index===0||testCase.id==='score-invalid') {
        await evaluate("document.querySelector('.detail-verdict').scrollIntoView({block:'start'})");
        const screenshot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
        fs.writeFileSync(path.join(out,'score-small-mobile-'+language+'-'+(index===0?'detail':'no-score')+'.png'),Buffer.from(screenshot.data,'base64'));
      }
      scoreChecks.push({language,id:testCase.id,list:row,detail});
    }
    await evaluate("document.getElementById('copy-json').click()");
    await wait('document.getElementById("copy-status").textContent.includes('+JSON.stringify(language==='zh'?'已复制':'copied')+')');
    assert(await evaluate('navigator.clipboard.readText()')===await evaluate('window.__scoreUiJson'),'Copying the score fixture changed original JSON in '+language);
  }
  await evaluate("window.FluxGlyphUI.setLanguage('zh')");
  const rejectionChecks=[];
  await evaluate(`window.__rejectionUiResult=${JSON.stringify(rejectionFixture)};window.__rejectionUiJson=JSON.stringify(window.__rejectionUiResult,null,2);window.FluxGlyphUI.show(window.__rejectionUiResult)`);
  for(const language of ['zh','en']) {
    await evaluate(`window.FluxGlyphUI.setLanguage(${JSON.stringify(language)});document.getElementById('close-detail').click();document.getElementById('regions').scrollIntoView({block:'start'})`);
    const state=await evaluate(`(() => ({accepted:document.querySelector('#summary .summary-chip.supported').textContent,
      rows:[...document.querySelectorAll('#regions .region')].map(row=>({id:row.dataset.regionId,
        prediction:row.querySelector('.font-prediction').textContent,score:row.querySelector('.font-score')?.textContent??null,
        status:row.querySelector('.tag').textContent,statusClass:row.querySelector('.tag').className,
        statusColor:getComputedStyle(row.querySelector('.tag')).color,overlayClass:document.querySelector('#overlay [data-region-id="'+row.dataset.regionId+'"]').getAttribute('class'),
        size:row.querySelector('.text-size')?.textContent??null,color:row.querySelector('.text-color').textContent,
        crop:row.querySelector('.region-thumbnail img').getAttribute('src'),description:row.querySelector('.region-description').textContent
      })),overflow:document.documentElement.scrollWidth>innerWidth
    }))()`);
    assert(state.accepted===(language==='zh'?'已识别 2':'Identified 2'),'Rejected predictions must not enter the accepted count, even with stale family/status fields');
    assert(!state.overflow,'Rejection list overflows 320px in '+language);
    const screen=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    fs.writeFileSync(path.join(out,'unknown-small-mobile-'+language+'-list.png'),Buffer.from(screen.data,'base64'));
    for(const [index,testCase] of rejectionUiCases.entries()) {
      const row=state.rows[index];const context=testCase.id+' / '+language;
      const known=testCase.kind==='known';
      const label=known?(language==='zh'?'最接近：':'Closest match: ')+testCase.family:
        testCase.kind==='unknown'?(language==='zh'?'未知字体':'Unknown font'):(language==='zh'?'未生成评分':'No model score');
      const score=known?(language==='zh'?'模型评分 ':'Model score ')+testCase.score.toFixed(4):null;
      const tag=testCase.status==='out_of_scope'?(language==='zh'?'字体未覆盖':'Font out of scope'):statusLabels[language][testCase.status];
      const color=testCase.status==='out_of_scope'?'rgb(107, 114, 128)':statusColors[testCase.status];
      assert(row.id===testCase.id&&row.prediction===label&&row.score===score,'Rejection/known list identity differs '+context);
      assert(row.status===tag&&row.statusClass==='tag '+testCase.status&&row.statusColor===color&&row.overlayClass.split(' ').includes(testCase.status),'Backend status or color changed '+context);
      assert(row.color.includes('#224466')&&row.crop==='/fixture/crop.png','Rejection must retain crop and visible color '+context);
      assert(known?row.size?.includes('42'):row.size===null,'Rejected/unavailable regions must have no displayed font size '+context);
      if(!known)assert(!/PingFang|SF Pro|0\.9998/.test(row.description),'Residual closed-set scores leaked a named font '+context);
      await evaluate('document.querySelector('+JSON.stringify('#regions [data-region-id="'+testCase.id+'"]')+').click()');
      await wait("document.querySelector('.detail-image').complete&&document.querySelector('.detail-image').naturalWidth>0");
      const detail=await evaluate(`(() => ({label:document.getElementById('font-label').textContent,
        score:document.getElementById('font-score')?.textContent??null,status:document.querySelector('.detail-verdict .tag').textContent,
        statusClass:document.querySelector('.detail-verdict .tag').className,
        top:[...document.querySelectorAll('.dist > span')].map(item=>item.textContent),
        reason:document.getElementById('font-reason').textContent,verdict:document.querySelector('.detail-verdict').textContent,
        size:document.querySelector('.text-style-detail .text-size')?.textContent??null,
        color:document.querySelector('.text-style-detail .text-color').textContent,
        crop:document.querySelector('.detail-image').getAttribute('src'),overflow:document.documentElement.scrollWidth>innerWidth,
        unchanged:document.getElementById('json-output').textContent===window.__rejectionUiJson&&JSON.stringify(window.__rejectionUiResult,null,2)===window.__rejectionUiJson
      }))()`);
      assert(detail.label===label&&detail.score===score&&detail.status===tag&&detail.statusClass===row.statusClass,'Rejection list/detail contract differs '+context);
      assert(detail.color.includes('#224466')&&detail.crop==='/fixture/crop.png','Rejection detail lost source/color '+context);
      assert(known?detail.size?.includes('42'):detail.size===null,'Rejection detail leaked font size '+context);
      if(known)assert(JSON.stringify(detail.top)===JSON.stringify(testCase.candidates.flatMap(item=>[item.family,item.score.toFixed(4)])),'Known/legacy Top 3 changed '+context);
      else {
        assert(detail.top.length===0&&!/PingFang|SF Pro|0\.9998/.test(detail.verdict),'Rejected/unavailable detail must suppress every named candidate/component '+context);
        const reason=testCase.kind==='unknown'?(language==='zh'?'当前模型无法识别':'does not recognize'):(language==='zh'?'判断结果异常':'invalid result');
        assert(detail.reason.includes(reason),'Unknown/invalid rejection reason must take priority '+context);
      }
      assert(!detail.overflow&&detail.unchanged,'Rejection UI overflow or raw JSON mutation '+context);
      if(index===1||index===4||index===6) {
        await evaluate("document.getElementById('detail-panel').scrollIntoView({block:'start'})");
        const screenshot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
        const kind=index===1?'detail':index===4?'invalid':'known';
        fs.writeFileSync(path.join(out,'unknown-small-mobile-'+language+'-'+kind+'.png'),Buffer.from(screenshot.data,'base64'));
      }
      rejectionChecks.push({language,id:testCase.id,list:row,detail});
    }
    await evaluate("document.getElementById('copy-json').click()");
    await wait('document.getElementById("copy-status").textContent.includes('+JSON.stringify(language==='zh'?'已复制':'copied')+')');
    assert(await evaluate('navigator.clipboard.readText()')===await evaluate('window.__rejectionUiJson'),'Rejection JSON clipboard changed '+language);
  }
  await evaluate("window.FluxGlyphUI.setLanguage('zh')");
  const consensusChecks=[];
  const verifierBase={method:'region_neural_network',status:'passed',family:'Helvetica',score:.94,
    candidates:[{family:'Helvetica',score:.94},{family:'SF Pro',score:.04},{family:'Roboto',score:.02}]};
  const consensusCases=[
    {id:'consensus-disagree',kind:'disagree',status:'uncertain',family:null,reason_code:'neural_model_disagreement',
      verifier:{...verifierBase,status:'disagreed'}},
    {id:'consensus-disagree-weak',kind:'disagree',status:'uncertain',family:null,reason_code:'neural_model_disagreement',
      verifier:{...verifierBase,status:'disagreed',score:.45,candidates:[{family:'Helvetica',score:.45},{family:'SF Pro',score:.44},{family:'Roboto',score:.11}]}},
    {id:'consensus-unknown',kind:'unknown',status:'out_of_scope',family:'SF Pro',reason_code:'verifier_font_out_of_scope',
      verifier:{...verifierBase,status:'out_of_scope',family:'Roboto',candidates:[{family:'Roboto',score:.99}]}},
    {id:'consensus-invalid',kind:'invalid',status:'uncertain',family:'SF Pro',reason_code:'invalid_verifier_output',
      verifier:{...verifierBase,status:'unavailable'}},
    {id:'consensus-agree',kind:'agree',status:'candidate',family:'SF Pro',reason_code:'region_neural_family_candidate',
      verifier:{...verifierBase,family:'SF Pro',candidates:[{family:'SF Pro',score:.94},{family:'Helvetica',score:.04},{family:'Roboto',score:.02}]}},
    {id:'consensus-low',kind:'low',status:'uncertain',family:null,reason_code:'verifier_below_score_gate',
      verifier:{...verifierBase,status:'below_gate',family:'SF Pro',score:.45,candidates:[{family:'SF Pro',score:.45},{family:'Helvetica',score:.44},{family:'Roboto',score:.11}]}}
  ];
  const consensusFixture={...fixture,id:'ui-consensus-only',regions:consensusCases.map(({id,kind,...font},i)=>({
    ...regions[i],id,font:{method:'region_neural_network',scope:'Detected text region',score:.9998,
      candidates:[{family:'SF Pro',score:.9998},{family:'Helvetica',score:.0002}],
      rejection:{status:'passed',known_score:.99,min_known_score:.8},...font},
    text_style:{font_size_px_estimate:kind==='low'?null:42,text_color_hex:'#224466',font_size_px_interval:null}
  }))};
  for(const [width,height] of [[1280,900],[320,700]]) for(const language of ['zh','en']) {
    await command('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
    await evaluate('window.FluxGlyphUI.setLanguage('+JSON.stringify(language)+');window.__consensusFixture='+JSON.stringify(consensusFixture)+';window.FluxGlyphUI.show(window.__consensusFixture);window.__consensusJson=document.getElementById("json-output").textContent;');
    for(const testCase of consensusCases) {
      await evaluate('document.querySelector('+JSON.stringify('#regions [data-region-id="'+testCase.id+'"]')+').click()');
      const state=await evaluate(`(() => {const row=document.querySelector('#regions .selected');return {
        row:row.textContent,prediction:row.querySelector('.font-prediction').textContent,label:document.getElementById('font-label').textContent,
        dual:[...document.querySelectorAll('.detail-verdict .font-consensus .font-score')].map(x=>x.textContent),
        tables:[...document.querySelectorAll('.dist')].map(x=>x.textContent),reason:document.getElementById('font-reason').textContent,
        rowSize:row.querySelector('.text-size')?.textContent??null,size:document.querySelector('.text-style-detail .text-size')?.textContent??null,
        color:document.querySelector('.text-style-detail .text-color').textContent,status:row.className,
        unchanged:document.getElementById('json-output').textContent===window.__consensusJson&&JSON.stringify(window.__consensusFixture,null,2)===window.__consensusJson,
        overflow:document.documentElement.scrollWidth>innerWidth};})()`);
      const context=testCase.id+'/'+width+'/'+language;
      assert(state.unchanged&&!state.overflow,'Consensus JSON mutation or overflow '+context);
      assert(state.color.includes('#224466')&&state.status.split(' ').includes(testCase.status),'Consensus lost status/color '+context);
      if(testCase.kind==='disagree') {
        assert(state.label===(language==='zh'?'字体存在分歧':'Font predictions disagree')&&state.prediction===state.label,'Disagreement needs prominent neutral label '+context);
        assert(!/最接近|Closest match/.test(state.row+state.label),'Single closest font still implies confirmation '+context);
        assert(state.dual.length===2&&state.dual[0].includes('SF Pro')&&state.dual[0].includes('0.9998')&&state.dual[1].includes('Helvetica')&&state.dual[1].includes(testCase.verifier.score.toFixed(4)),'Both networks need distinct scores '+context);
        assert(state.tables.length===2&&state.tables[0].includes('0.9998')&&state.tables[1].includes(testCase.verifier.score.toFixed(4)),'Both Top 3 tables missing '+context);
        assert(state.rowSize===null&&state.size===null,'Disagreement must suppress stale size '+context);
        if(testCase.id==='consensus-disagree') {
          await evaluate("document.getElementById('detail-panel').scrollIntoView({block:'start'})");
          const shot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
          fs.writeFileSync(path.join(out,'consensus-'+width+'-'+language+'.png'),Buffer.from(shot.data,'base64'));
        }
      } else if(testCase.kind==='unknown'||testCase.kind==='invalid') {
        const label=testCase.kind==='unknown'?(language==='zh'?'未知字体':'Unknown font'):(language==='zh'?'未生成评分':'No model score');
        assert(state.label===label&&state.prediction===label,'Unknown/invalid verifier label differs '+context);
        assert(!/SF Pro|Helvetica|Roboto|0\.9998/.test(state.row)&&state.tables.length===0&&state.dual.length===0,'Unknown/invalid verifier leaked named scores '+context);
        assert(state.rowSize===null&&state.size===null,'Unknown/invalid verifier leaked size '+context);
      } else {
        assert(state.label.includes('SF Pro')&&state.tables.length===2,'Passed/low verifier scores missing '+context);
        if(testCase.kind==='agree')assert(state.size.includes('42'),'Agreement lost primary size '+context);
        else assert(state.reason.includes(language==='zh'?'复核网络':'verifier')&&!state.size.includes('42'),'Verifier gate reason or null size missing '+context);
      }
      consensusChecks.push({id:testCase.id,width,language,...state});
    }
  }
  await evaluate("window.FluxGlyphUI.setLanguage('zh')");
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

  const modeChecks=[];
  for(const language of ['zh','en']) {
    await evaluate(`window.FluxGlyphUI.setLanguage('${language}');document.querySelector('[data-font-mode="android"]').click()`);
    await wait("document.getElementById('download-model').getAttribute('href')==='/api/models/font/download?mode=android'");
    const state=await evaluate(`({mode:window.FluxGlyphUI.get().fontMode,info:document.getElementById('model-info').textContent,note:document.getElementById('model-label-note').textContent,help:document.querySelector('[data-i18n="fontModeHelp"]').textContent,overflow:document.documentElement.scrollWidth>innerWidth,resultHidden:document.getElementById('result-section').hidden,previewHidden:document.getElementById('android-preview').hidden,preview:document.getElementById('android-preview').textContent,previewLink:document.querySelector('#android-preview a').getAttribute('href'),previewBeforeUpload:document.getElementById('android-preview').getBoundingClientRect().bottom<=document.querySelector('.upload-help').getBoundingClientRect().top,downloadLabel:document.getElementById('download-model').textContent})`);
    assert(state.mode==='android'&&state.info.includes('android-ui-fixture')&&state.resultHidden,'Mode switch must clear prior result and use independent download');
    assert(state.note.includes('Source Han Sans')&&!state.note.includes('PingFang')&&!state.overflow,'Android alias group must remain visible and fit mobile');
    assert(state.help.includes(language==='zh'?'不用于判断':'does not identify'),'Mode must not claim device OS detection');
    assert(!state.previewHidden&&state.previewBeforeUpload&&state.previewLink==='/docs#android-preview','Experimental notice must be visible before upload and link to its evaluation');
    assert(state.preview.includes(language==='zh'?'尚未通过稳定版验收':'Stable validation has not passed')&&state.preview.includes(language==='zh'?'分数不是准确率':'Scores are not accuracy'),'Preview must state stable failure and score limits in both languages');
    assert(state.downloadLabel.includes(language==='zh'?'实验模型':'preview'),'Android experimental download must be labelled');
    assert(await evaluate("fetch(document.getElementById('download-model').href).then(r=>r.text())")==='Independent Android UI download fixture','Android link fetched the wrong model');
    modeChecks.push({language,...state});
    await evaluate("window.FluxGlyphUI.setFontMode('ios')");
    await wait("document.getElementById('download-model').getAttribute('href')==='/api/models/font/download'");
    assert(await evaluate("document.getElementById('android-preview').hidden"),'Android preview warning leaked into iOS mode');
  }
  androidRelease={};
  await evaluate("window.FluxGlyphUI.setFontMode('android')");
  await wait("document.getElementById('download-model').getAttribute('href')==='/api/models/font/download?mode=android'");
  assert(await evaluate("document.getElementById('android-preview').hidden && !document.getElementById('download-model').textContent.includes('preview')"),'Old Android metadata must not invent release validation');
  androidRelease=previewMetadata;
  await evaluate("window.FluxGlyphUI.setFontMode('ios')");
  await wait("document.getElementById('download-model').getAttribute('href')==='/api/models/font/download'");
  await evaluate("window.FluxGlyphUI.setLanguage('zh')");
  androidAvailability=false;
  await evaluate("window.FluxGlyphUI.setFontMode('android');fetch('/fixture/crop.png').then(r=>r.blob()).then(blob=>window.FluxGlyphUI.setFile(new File([blob],'android-ui.png',{type:'image/png'})))");
  await wait("document.getElementById('font-model').getAttribute('aria-busy')==='false'");
  assert(await evaluate("document.getElementById('start').disabled && !document.getElementById('download-model').hasAttribute('href') && document.getElementById('model-info').textContent.includes('安卓字体模型暂不可用')"),'Missing Android must disable prediction and download, even after selecting a file');
  await evaluate("window.FluxGlyphUI.start('/api/jobs',{method:'POST',body:new Uint8Array([1])})");
  assert(modeUploads.length===0,'Missing Android must not silently upload to iOS');
  androidAvailability=true;
  await evaluate("document.getElementById('refresh-model').click()");
  await wait("!document.getElementById('start').disabled");
  await evaluate("document.getElementById('start').click()");
  await wait("window.FluxGlyphUI.get().snapshot?.status==='queued'");
  assert(modeUploads.length===1&&modeUploads[0].bytes===cropPng.length,'Real mock upload did not carry selected Android mode and PNG bytes');
  assert(await evaluate("[...document.querySelectorAll('[data-font-mode]')].every(b=>b.disabled)"),'Mode must be locked while a request is active');
  await evaluate("window.FluxGlyphUI.setFontMode('ios')");
  assert(await evaluate("window.FluxGlyphUI.get().fontMode==='android'"),'Active job changed model scope');
  androidJobComplete=true;
  await wait("window.FluxGlyphUI.get().result?.font_mode==='android'");
  const completed=await evaluate("({json:JSON.parse(document.getElementById('json-output').textContent),row:document.querySelector('#regions .region').textContent,unlocked:[...document.querySelectorAll('[data-font-mode]')].every(b=>!b.disabled)})");
  assert(completed.json.font_mode==='android'&&completed.json.device_inference_performed===false&&completed.unlocked,'Completed result lost scope or mode remained locked');
  assert(completed.json.model_release_tier==='experimental'&&completed.json.stable_validation_passed===false&&JSON.stringify(completed.json)===JSON.stringify(androidFixture),'JSON must retain failed preview evidence unchanged');
  assert(completed.row.includes('Noto Sans CJK SC')&&completed.row.includes('0.8765')&&!completed.row.includes('PingFang'),'Android result must present its own classifier scores');
  modeChecks.push({completed:true,...completed});
  await evaluate("window.scrollTo(0,0)");
  fs.writeFileSync(path.join(out,'android-mode-mobile.png'),Buffer.from((await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false})).data,'base64'));
  await evaluate("window.FluxGlyphUI.setFontMode('ios')");
  await wait("document.getElementById('download-model').getAttribute('href')==='/api/models/font/download'");
  androidDelay=250;
  await evaluate("window.FluxGlyphUI.setFontMode('android')");
  await sleep(40);
  await evaluate("window.FluxGlyphUI.setFontMode('ios')");
  await sleep(350);
  assert(await evaluate("window.FluxGlyphUI.get().fontMode==='ios' && document.getElementById('download-model').getAttribute('href')==='/api/models/font/download' && !document.getElementById('model-info').textContent.includes('android')"),'Stale Android response replaced iOS download after switching back');
  modeChecks.push({stale_response_ignored:true,unavailable_no_fallback:true,queue_model_pinned:true});

  assert(!diagnostics.length,'JS exceptions '+diagnostics.join());
  fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({kind:'mocked region-result browser UI only, not model accuracy',passed:true,reports,scoreChecks,rejectionChecks,consensusChecks,availabilityChecks,sourceCompatibilityChecks,modeChecks,diagnostics},null,2));
  console.log(JSON.stringify({passed:true,viewports:reports.map(x=>x.name),output:out}));
 } finally {if(socket)socket.close();chrome.kill('SIGTERM');await new Promise(resolve=>chrome.exitCode!==null?resolve():chrome.once('exit',resolve));server.close();fs.rmSync(profile,{recursive:true,force:true});}
})().catch(error=>{console.error(error);server.close();process.exitCode=1;});
