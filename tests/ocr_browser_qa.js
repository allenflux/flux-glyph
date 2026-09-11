'use strict';

const assert = require('assert');
const {spawn, spawnSync} = require('child_process');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const BASE_URL = process.env.FLUX_GLYPH_URL || 'http://127.0.0.1:9000';
const SOURCE = path.resolve('tests/fixtures/ui_title_billing_details.png');
const OUTPUT = path.resolve('docs/ui-validation');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function request(url) {
  return new Promise((resolve, reject) => {
    http.get(url, response => {
      let body = '';
      response.on('data', chunk => { body += chunk; });
      response.on('end', () => response.statusCode >= 400 ? reject(new Error(`HTTP ${response.statusCode}: ${body}`)) : resolve(body));
    }).on('error', reject);
  });
}

async function waitForPort(profile) {
  const file = path.join(profile, 'DevToolsActivePort');
  for (let i = 0; i < 150; i += 1) {
    if (fs.existsSync(file)) return Number(fs.readFileSync(file, 'utf8').split('\n')[0]);
    await sleep(100);
  }
  throw new Error('Chrome did not expose a debugging port');
}

function cdp(url) {
  const socket = new WebSocket(url);
  const pending = new Map();
  const errors = [];
  let id = 0;
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const item = pending.get(message.id); pending.delete(message.id);
      message.error ? item.reject(new Error(message.error.message)) : item.resolve(message.result);
    } else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
    else if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') errors.push(message.params.entry.text);
  };
  const opened = new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  const command = (method, params = {}) => new Promise((resolve, reject) => {
    const key = ++id; pending.set(key, {resolve, reject}); socket.send(JSON.stringify({id: key, method, params}));
    setTimeout(() => { if (pending.delete(key)) reject(new Error(`CDP timeout: ${method}`)); }, 30000);
  });
  return {socket, opened, command, errors};
}

async function evaluate(command, expression, awaitPromise = false, userGesture = false) {
  const answer = await command('Runtime.evaluate', {expression, awaitPromise, userGesture, returnByValue: true});
  if (answer.exceptionDetails) throw new Error(answer.exceptionDetails.exception?.description || answer.exceptionDetails.text);
  return answer.result.value;
}

async function waitFor(command, expression, label, timeout = 240000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    const value = await evaluate(command, expression);
    if (value) return value;
    await sleep(150);
  }
  throw new Error(`Timed out waiting for ${label}`);
}

function makeLongFixture(target) {
  const python = fs.existsSync('.venv-dev/bin/python') ? '.venv-dev/bin/python' : 'python3';
  const code = [
    'from PIL import Image, ImageDraw', 'import sys',
    'tile=Image.open(sys.argv[1]).convert("RGB")',
    'canvas=Image.new("RGB",(760,980),"white")',
    'd=ImageDraw.Draw(canvas)',
    'for x in (38,265,492): canvas.paste(tile,(x,70))',
    'for x in (90,317): canvas.paste(tile,(x,330))',
    'canvas.paste(tile,(430,570))',
    'font_path="/System/Library/Fonts/Supplemental/Arial.ttf"',
    'from PIL import ImageFont',
    'font=ImageFont.truetype(font_path,42)',
    'd.text((55,790),"ORDER 2026 / TOTAL 88.50",fill="#202020",font=font)',
    'canvas.save(sys.argv[2],"PNG",optimize=True)',
  ].join('\n');
  const result = spawnSync(python, ['-c', code, SOURCE, target], {encoding: 'utf8'});
  assert.strictEqual(result.status, 0, result.stderr || 'fixture generation failed');
}

async function screenshot(command, name) {
  const result = await command('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false});
  const file = path.join(OUTPUT, name); fs.writeFileSync(file, Buffer.from(result.data, 'base64')); return path.relative(process.cwd(), file);
}

async function layout(command, width, height, language) {
  await command('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor: 1, mobile: width <= 390});
  const value = await evaluate(command, `(() => {
    document.querySelector('[data-language="${language}"]').click();
    document.querySelector('.viewer').scrollTop=0;document.getElementById('regions').scrollTop=0;document.getElementById('ocr-output').scrollTop=0;
    const q=s=>document.querySelector(s), r=e=>{const x=e.getBoundingClientRect();return {x:x.x,y:x.y,w:x.width,h:x.height,b:x.bottom,r:x.right}};
    const visual=r(q('.result-visual')), ocr=r(q('.ocr-panel')), viewer=r(q('.viewer')), image=r(q('.image-layer')), svg=r(q('#overlay')), original=r(q('#original')), upload=r(q('.upload-help'));
    return {lang:document.documentElement.lang,scrollWidth:document.documentElement.scrollWidth,innerWidth,
      visual,ocr,viewer,image,svg,original,overlayAligned:Math.abs(svg.x-original.x)<.5&&Math.abs(svg.y-original.y)<.5&&Math.abs(svg.w-original.w)<.5&&Math.abs(svg.h-original.h)<.5,upload,viewport:{w:innerWidth,h:innerHeight},ocrRows:q('#ocr-output').querySelectorAll('.ocr-row').length,
      ocrText:q('#ocr-output').innerText,copyText:q('#copy-ocr').textContent,resultTop:r(q('#result-section')).y,
      apiButton:q('#api-link').getBoundingClientRect().height, imageReasonable:viewer.h <= innerHeight*.82,
      offenders:[...document.querySelectorAll('body *')].map(e=>({tag:e.tagName,id:e.id,cls:e.className?.baseVal||e.className||'',r:e.getBoundingClientRect().right,w:e.getBoundingClientRect().width,scroll:e.scrollWidth})).filter(x=>x.r>innerWidth+1),
      mobileStacked:ocr.y >= visual.b-1, desktopAdjacent:ocr.x >= visual.r-1 && ocr.y < visual.b,
      selected:[...q('#ocr-output').querySelectorAll('.ocr-row.selected')].length};
  })()`);
  assert(value.scrollWidth <= value.innerWidth + 1, `${width}px horizontal overflow: ${JSON.stringify(value)}`);
  assert(value.ocrRows >= 2 && value.ocrText.trim(), `${width}px OCR transcript is missing`);
  assert(value.apiButton >= 32, `${width}px API tool button is too small`);
  assert(value.imageReasonable, `${width}px result image overwhelms viewport: ${JSON.stringify(value.image)}`);
  assert(value.overlayAligned, `${width}px source image and SVG overlay diverged: ${JSON.stringify({image:value.original,svg:value.svg})}`);
  assert(width <= 1180 ? value.mobileStacked : value.desktopAdjacent, `${width}px result layout has the wrong proportion: ${JSON.stringify(value)}`);
  await evaluate(command, "window.scrollTo(0,document.getElementById('result-section').offsetTop);true");
  const resultShot = await screenshot(command, `ocr-${width}x${height}-${language}.png`);
  let topShot = null;
  if (width === 1920 && height === 900 && language === 'zh') { await evaluate(command, 'window.scrollTo(0,0);true'); topShot=await screenshot(command,'ocr-1920x900-zh-page-top.png'); }
  return {...value, screenshot:resultShot, topScreenshot:topShot};
}

(async () => {
  [CHROME, SOURCE].forEach(file => assert(fs.existsSync(file), `Missing ${file}`));
  fs.mkdirSync(OUTPUT, {recursive: true});
  await request(`${BASE_URL}/api/health`);
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'flux-glyph-ocr-'));
  const fixture = path.join(temp, 'ocr-multi-region-long.png'); makeLongFixture(fixture);
  const downloads = path.join(temp, 'downloads'); fs.mkdirSync(downloads);
  const chrome = spawn(CHROME, ['--headless=new', '--disable-gpu', '--no-first-run', '--remote-debugging-port=0', `--user-data-dir=${path.join(temp, 'profile')}`, BASE_URL], {stdio: 'ignore'});
  try {
    const port = await waitForPort(path.join(temp, 'profile'));
    const targets = JSON.parse(await request(`http://127.0.0.1:${port}/json/list`));
    const client = cdp(targets.find(item => item.type === 'page').webSocketDebuggerUrl); await client.opened;
    const {command} = client;
    await Promise.all(['Page.enable','Runtime.enable','Log.enable','DOM.enable','Network.enable'].map(name => command(name)));
    await command('Browser.grantPermissions', {origin: BASE_URL, permissions: ['clipboardReadWrite', 'clipboardSanitizedWrite']});
    await command('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath: downloads});
    await waitFor(command, "document.readyState==='complete' && !!window.FluxGlyphUI", 'app');
    const doc = await command('DOM.getDocument');
    const input = await command('DOM.querySelector', {nodeId: doc.root.nodeId, selector: '#file'});
    await command('DOM.setFileInputFiles', {nodeId: input.nodeId, files: [fixture]});
    await evaluate(command, "document.getElementById('start').click(); true");
    const timeline = [];
    const end = Date.now() + 240000;
    while (Date.now() < end) {
      const state = await evaluate(command, `(() => {const s=window.FluxGlyphUI.get(),p=document.getElementById('progress-panel'),t=document.getElementById('progress-track'),b=document.getElementById('progress-bar');return {done:!!s.result,error:document.getElementById('error').hidden?'':document.getElementById('error').textContent,stage:p.dataset.stage||t.dataset.stage,status:s.snapshot?.status||'',color:getComputedStyle(b).backgroundColor,panel:getComputedStyle(p).backgroundColor}})()`);
      if (!timeline.length || JSON.stringify(state) !== JSON.stringify(timeline[timeline.length-1])) timeline.push(state);
      if (state.error) throw new Error(state.error);
      if (state.done) break;
      await sleep(100);
    }
    assert(timeline.at(-1)?.done, 'real OCR did not complete');
    assert(timeline.some(x => x.stage === 'running' || ['preparing','detecting','recognizing','matching','annotating','finalizing'].includes(x.status)), `running stage not observed: ${JSON.stringify(timeline)}`);
    assert(timeline.at(-1).stage === 'complete', `complete stage not exposed: ${JSON.stringify(timeline.at(-1))}`);
    const settledComplete = await waitFor(command, `(() => {const p=document.getElementById('progress-panel'),c=getComputedStyle(document.getElementById('progress-bar')).backgroundColor;return p.dataset.stage==='complete'&&c==='rgb(20, 128, 74)'&&{stage:p.dataset.stage,color:c};})()`, 'settled green completion color', 3000);
    const colors = Object.fromEntries(timeline.filter(x => x.stage).map(x => [x.stage, x.color])); colors.complete = settledComplete.color;
    assert(colors.running === 'rgb(37, 99, 235)' && colors.complete === 'rgb(20, 128, 74)', `running/completion colors are incorrect: ${JSON.stringify(colors)}`);
    await waitFor(command, "document.querySelectorAll('#ocr-output .ocr-row').length>=2 && document.querySelector('#overlay polygon')", 'OCR transcript');

    const linkage = await evaluate(command, `(() => {
      const rows=[...document.querySelectorAll('#ocr-output .ocr-row')], viewer=document.querySelector('.viewer'), list=document.getElementById('regions'), detail=document.getElementById('detail');
      window.scrollTo(0,document.getElementById('result-section').offsetTop); const pageY=window.scrollY, listTop=list.getBoundingClientRect().top;
      const checks=[]; for (const target of [rows[0],rows[Math.floor(rows.length/2)],rows[rows.length-1]]) {const beforeY=window.scrollY;target.click();checks.push({id:target.dataset.regionId,selected:String(window.FluxGlyphUI.get().selected),row:target.classList.contains('selected'),polygon:!!document.querySelector('#overlay polygon.selected'),region:!!document.querySelector('#regions .region.selected'),pageStable:Math.abs(window.scrollY-beforeY)<1,listStable:Math.abs(list.getBoundingClientRect().top-listTop)<1,viewerScroll:viewer.scrollTop,detailHeight:detail.getBoundingClientRect().height,detailMax:getComputedStyle(detail).maxHeight});}
      const polygon=document.querySelectorAll('#overlay polygon')[0], beforePolygonY=window.scrollY, beforeViewer=viewer.scrollTop;polygon.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true}));
      return {checks,pageY,polygonPageStable:Math.abs(window.scrollY-beforePolygonY)<1,polygonViewerStable:Math.abs(viewer.scrollTop-beforeViewer)<1,detail:detail.innerText,detailScrolls:detail.scrollHeight<=detail.clientHeight+1||['auto','scroll'].includes(getComputedStyle(detail).overflowY)};
    })()`);
    assert(linkage.checks.every(x => x.selected === x.id && x.row && x.polygon && x.region && x.pageStable && x.listStable), `OCR row linkage or stable-page layout failed: ${JSON.stringify(linkage)}`);
    assert(linkage.polygonPageStable && linkage.polygonViewerStable && linkage.detail.length > 10 && linkage.detailScrolls, `Polygon selection moved the page/viewer or detail is unbounded: ${JSON.stringify(linkage)}`);

    const exact = await evaluate(command, `(() => {const r=window.FluxGlyphUI.get().result;return {ocr:r.regions.map(x=>x.text).filter(Boolean).join('\\n'),json:JSON.stringify(r,null,2),shown:document.getElementById('ocr-output').innerText};})()`);
    await evaluate(command, "document.getElementById('copy-ocr').click(); true", false, true);
    await waitFor(command, "document.getElementById('ocr-copy-status').textContent.trim()", 'OCR clipboard confirmation');
    const clipboardOcr = await evaluate(command, 'navigator.clipboard.readText()', true);
    assert.strictEqual(clipboardOcr, exact.ocr, 'Clipboard API did not copy exact OCR transcript');
    await evaluate(command, "Object.defineProperty(navigator,'clipboard',{configurable:true,value:undefined});document.getElementById('copy-ocr').click();true", false, true);
    await sleep(100); await evaluate(command, 'delete navigator.clipboard;true');
    assert.strictEqual(await evaluate(command, 'navigator.clipboard.readText()', true), exact.ocr, 'OCR fallback copy was not exact');
    await evaluate(command, "document.getElementById('copy-json').click();true", false, true); await sleep(100);
    assert.strictEqual(await evaluate(command, 'navigator.clipboard.readText()', true), exact.json, 'JSON copy changed');

    const before = new Set(fs.readdirSync(downloads));
    await evaluate(command, "document.getElementById('download-png').click();true", false, true);
    await waitFor(command, `false`, 'download', 200).catch(() => {});
    for (let i=0;i<100 && !fs.readdirSync(downloads).some(x=>!before.has(x)&&!x.endsWith('.crdownload'));i+=1) await sleep(100);
    const png = fs.readdirSync(downloads).find(x => !before.has(x) && !x.endsWith('.crdownload'));
    assert(png && fs.readFileSync(path.join(downloads,png)).subarray(1,4).equals(Buffer.from('PNG')), 'annotated PNG download failed');

    const layouts = [];
    for (const [w,h] of [[1920,900],[1920,1080],[1280,900],[768,900],[740,900],[390,844],[320,700]]) {
      layouts.push(await layout(command,w,h,'zh')); layouts.push(await layout(command,w,h,'en'));
    }
    assert(await evaluate(command, `document.getElementById('json-output').textContent===${JSON.stringify(exact.json)}`), 'language/viewport changes mutated JSON');

    const invalid = path.join(temp, 'invalid.png'); fs.writeFileSync(invalid, Buffer.from('not a png'));
    await command('DOM.setFileInputFiles', {nodeId: input.nodeId, files: [invalid]});
    await evaluate(command, "document.getElementById('start').click();true");
    const errorState = await waitFor(command, `(() => {const p=document.getElementById('progress-panel'),label=document.getElementById('progress-label'),track=document.getElementById('progress-track'),c=getComputedStyle(label).color;return p.dataset.stage==='error'&&c==='rgb(180, 35, 24)'&&{stage:p.dataset.stage,color:c,trackColor:getComputedStyle(track).backgroundColor,ariaNow:track.getAttribute('aria-valuenow'),message:document.getElementById('error').textContent,resultHidden:document.getElementById('result-section').hidden};})()`, 'settled red error state', 10000);
    assert(errorState.ariaNow === null, `error progress retained a misleading numeric value: ${JSON.stringify(errorState)}`);
    assert(errorState.message && errorState.resultHidden, `real invalid upload did not clear stale output: ${JSON.stringify(errorState)}`); colors.error=errorState.color;
    const reset = await evaluate(command, `(() => {const f=document.getElementById('file');f.value='';f.dispatchEvent(new Event('change',{bubbles:true}));const p=document.getElementById('progress-panel');return {hidden:p.hidden,stage:p.dataset.stage||'',bar:document.getElementById('progress-bar').style.width,result:document.getElementById('result-section').hidden,ocr:document.getElementById('ocr-output').innerText};})()`);
    assert(reset.hidden && !reset.stage && reset.result && !reset.ocr.trim(), `progress/result reset failed: ${JSON.stringify(reset)}`);

    const docs = [];
    await command('Page.navigate', {url: `${BASE_URL}/docs`});
    await waitFor(command, "document.readyState==='complete' && !!document.getElementById('language')", 'API docs');
    for (const [w,h] of [[390,844],[320,700]]) {
      await command('Emulation.setDeviceMetricsOverride', {width:w,height:h,deviceScaleFactor:1,mobile:true});
      for (const language of ['zh','en']) {
        const item = await evaluate(command, `(() => {document.querySelector('[data-language="${language}"]').click();const h=document.querySelector('.page-heading').getBoundingClientRect(),tools=document.querySelector('.header-tools').getBoundingClientRect(),title=document.querySelector('.page-heading h1').getBoundingClientRect();return {width:${w},language,scrollWidth:document.documentElement.scrollWidth,innerWidth,tools:{x:tools.x,y:tools.y,w:tools.width,h:tools.height},title:{x:title.x,y:title.y,w:title.width,h:title.height},header:{x:h.x,y:h.y,w:h.width,h:h.height},lang:document.documentElement.lang};})()`);
        assert(item.scrollWidth <= item.innerWidth + 1, `docs ${w}px ${language} overflows: ${JSON.stringify(item)}`);
        const overlaps = !(item.tools.x >= item.title.x + item.title.w || item.title.x >= item.tools.x + item.tools.w || item.tools.y >= item.title.y + item.title.h || item.title.y >= item.tools.y + item.tools.h);
        assert(!overlaps, `docs ${w}px ${language} header overlaps: ${JSON.stringify(item)}`);
        docs.push(item);
      }
    }

    const browserErrors = client.errors.filter(message => !/status of 400 \(Bad Request\)/.test(message));
    const report = {result:'passed',testedAt:new Date().toISOString(),fixture:'synthetic multi-region composition of tests/fixtures/ui_title_billing_details.png plus rendered Latin/digits',timeline,colors,regions:exact.ocr.split('\n').length,linkage,clipboard:{api:true,fallback:true,jsonExact:true},png,reset,layouts,docs,browserErrors,ignoredDiagnostics:client.errors.filter(message => !browserErrors.includes(message))};
    assert(!browserErrors.length, `browser errors: ${browserErrors.join(' | ')}`);
    fs.writeFileSync(path.join(OUTPUT,'ocr-browser-qa.json'), `${JSON.stringify(report,null,2)}\n`);
    console.log(JSON.stringify(report,null,2)); client.socket.close();
  } finally {
    chrome.kill('SIGTERM');
  }
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
