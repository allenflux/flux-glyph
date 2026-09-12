'use strict';

const {spawn, spawnSync} = require('child_process');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const BASE_URL = process.env.FLUX_GLYPH_URL || 'http://127.0.0.1:9000';
// Optional deployment pin; the tested version always comes from live health.
const PINNED_MODEL = process.env.FLUX_EXPECTED_MODEL || null;
const FIXTURE = path.resolve('tests/fixtures/ui_title_billing_details.png');
const OUTPUT = path.resolve(process.env.FLUX_QA_OUTPUT || 'docs/ui-validation');
const CASES = [
  {name: 'desktop', width: 1280, height: 900, mobile: false, forceQueue: true},
  {name: 'mobile', width: 390, height: 844, mobile: true, forceQueue: false},
];
const PROGRESS_STAGES = new Set(['queued', 'preparing', 'detecting', 'recognizing', 'matching', 'annotating', 'finalizing', 'complete', 'error']);

const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
const assert = (condition, message) => { if (!condition) throw new Error(message); };
let queueSource = null;

function createSyntheticQueueFixture() {
  const output = path.join(os.tmpdir(), `flux-glyph-queue-${process.pid}.png`);
  const pythonCandidates = [process.env.FLUX_PYTHON, path.resolve('.venv-dev/bin/python'), path.resolve('.venv/bin/python'), 'python3'].filter(Boolean);
  const python = pythonCandidates.find(candidate => candidate === 'python3' || fs.existsSync(candidate));
  const script = [
    'from PIL import Image',
    'import sys',
    'tile=Image.open(sys.argv[1]).convert("RGB")',
    'canvas=Image.new("RGB",(750,1120),"white")',
    'xs=(14,261,508)',
    'for row in range(14):',
    '    for x in xs: canvas.paste(tile,(x,10+row*79))',
    'canvas.save(sys.argv[2],"PNG",optimize=True)',
  ].join('\n');
  const generated = spawnSync(python, ['-c', script, FIXTURE, output], {encoding: 'utf8'});
  assert(generated.status === 0 && fs.existsSync(output), `Could not generate synthetic queue fixture: ${generated.stderr || generated.stdout}`);
  return output;
}

function requestJson(url, options = {}) {
  return new Promise((resolve, reject) => {
    const request = http.request(new URL(url), {method: options.method || 'GET', headers: options.headers || {}}, response => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', chunk => { body += chunk; });
      response.on('end', () => {
        try {
          const data = JSON.parse(body);
          if (response.statusCode >= 400) reject(new Error(`HTTP ${response.statusCode}: ${body}`));
          else resolve({status: response.statusCode, data});
        } catch (error) { reject(error); }
      });
    });
    request.on('error', reject);
    request.end(options.body);
  });
}

async function submitQueueFillers(count) {
  const body = fs.readFileSync(queueSource);
  const responses = [];
  for (let index = 0; index < count; index += 1) {
    const response = await requestJson(`${BASE_URL}/api/jobs`, {
      method: 'POST',
      headers: {'Content-Type': 'image/png', 'Content-Length': body.length, 'X-Filename': `qa-synthetic-queue-${index + 1}.png`},
      body,
    });
    assert(response.status === 202, `Queue filler returned HTTP ${response.status}`);
    responses.push(response);
  }
  return responses.map(response => response.data.id);
}

async function waitForServerIdle(timeoutMilliseconds = 240000) {
  const deadline = Date.now() + timeoutMilliseconds;
  while (Date.now() < deadline) {
    const health = (await requestJson(`${BASE_URL}/api/health`)).data;
    if (health.running_jobs === 0 && health.waiting_jobs === 0) return health;
    await sleep(500);
  }
  throw new Error('Timed out waiting for the inference queue to become idle');
}

async function waitForDebuggingPort(profileDirectory) {
  const portFile = path.join(profileDirectory, 'DevToolsActivePort');
  for (let attempt = 0; attempt < 120; attempt += 1) {
    if (fs.existsSync(portFile)) {
      const [port] = fs.readFileSync(portFile, 'utf8').trim().split('\n');
      if (port) return Number(port);
    }
    await sleep(100);
  }
  throw new Error('Chrome did not expose a DevTools port');
}

async function waitForPage(port) {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    try {
      const response = await requestJson(`http://127.0.0.1:${port}/json/list`);
      const page = response.data.find(target => target.type === 'page');
      if (page) return page;
    } catch (_) { /* The port can appear before its target endpoint is ready. */ }
    await sleep(100);
  }
  throw new Error('Chrome did not create a page target');
}

function connect(webSocketDebuggerUrl) {
  const socket = new WebSocket(webSocketDebuggerUrl);
  let sequence = 0;
  const pending = new Map();
  const diagnostics = [];
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const {resolve, reject} = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) reject(new Error(message.error.message)); else resolve(message.result);
      return;
    }
    if (message.method === 'Runtime.exceptionThrown') diagnostics.push(`exception: ${message.params.exceptionDetails.text}`);
    if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') {
      const entry = message.params.entry;
      diagnostics.push(`console: ${entry.text}${entry.url ? ` (${entry.url})` : ''}`);
    }
    if (message.method === 'Network.responseReceived' && message.params.response.status >= 400) {
      const response = message.params.response;
      diagnostics.push(`http ${response.status}: ${response.url}`);
    }
  };
  const opened = new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  const command = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    const timeout = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`CDP command timed out: ${method}`));
    }, 30000);
    pending.set(id, {
      resolve: value => { clearTimeout(timeout); resolve(value); },
      reject: error => { clearTimeout(timeout); reject(error); },
    });
    socket.send(JSON.stringify({id, method, params}));
  });
  return {socket, opened, command, diagnostics};
}

async function evaluate(command, expression, awaitPromise = false, userGesture = false) {
  const response = await command('Runtime.evaluate', {expression, awaitPromise, userGesture, returnByValue: true});
  if (response.exceptionDetails) throw new Error(response.exceptionDetails.exception?.description || response.exceptionDetails.text);
  return response.result.value;
}

async function waitFor(command, expression, description, timeoutMilliseconds = 240000, intervalMilliseconds = 250) {
  const deadline = Date.now() + timeoutMilliseconds;
  let value;
  while (Date.now() < deadline) {
    value = await evaluate(command, expression);
    if (value) return value;
    await sleep(intervalMilliseconds);
  }
  throw new Error(`Timed out waiting for ${description}; last value: ${JSON.stringify(value)}`);
}

async function waitForDownload(directory, previousNames, description) {
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    const fresh = fs.readdirSync(directory).filter(name => !previousNames.has(name));
    const complete = fresh.find(name => !name.endsWith('.crdownload'));
    if (complete && !fresh.some(name => name.endsWith('.crdownload'))) return path.join(directory, complete);
    await sleep(200);
  }
  throw new Error(`Timed out waiting for ${description}`);
}

function pngDimensions(buffer) {
  assert(buffer.length >= 24 && buffer.subarray(1, 4).equals(Buffer.from('PNG')), 'Invalid PNG signature');
  return {width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20)};
}

async function captureScreenshot(command, filename) {
  const shot = await command('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
  const output = path.join(OUTPUT, filename);
  fs.writeFileSync(output, Buffer.from(shot.data, 'base64'));
  return path.relative(process.cwd(), output);
}

async function switchLanguage(command, language) {
  return evaluate(command, `(() => {
    const control = document.getElementById('language');
    control.querySelector('[data-language="${language}"]').click();
    return {
      value: control.dataset.language,
      pressed: Object.fromEntries([...control.querySelectorAll('[data-language]')].map(button => [button.dataset.language, button.getAttribute('aria-pressed')])),
      lang: document.documentElement.lang,
      uploadTitle: document.getElementById('upload-title').textContent,
      start: document.getElementById('start').textContent,
      resultTitle: document.querySelector('#result-section .panel-heading h2').textContent,
      copyJson: document.getElementById('copy-json').textContent,
      downloadPng: document.getElementById('download-png').textContent,
      status: document.getElementById('status').textContent,
      progressLabel: document.getElementById('progress-label').textContent,
      progressAria: document.getElementById('progress-track').getAttribute('aria-valuetext'),
      queueStatus: document.getElementById('queue-status').textContent,
      fontReason: document.getElementById('font-reason')?.textContent || '',
      regionTitle: document.querySelector('.detail-preview h2')?.textContent || '',
      regionCrop: Boolean(document.querySelector('.detail-preview .detail-image')),
      fontSources: [...document.querySelectorAll('#model-families p')].map(row => ({source: row.dataset.fontSource || null, text: row.textContent})),
      fontLabelNote: {hidden: document.getElementById('model-label-note')?.hidden !== false, text: document.getElementById('model-label-note')?.textContent || ''},
      json: document.getElementById('json-output').textContent,
      ocr: (window.FluxGlyphUI.get().result?.regions || []).map(region => region.text),
    };
  })()`);
}

function assertFontSourceLabels(info, snapshot, language) {
  const families = info.families.filter(family => typeof family === 'string');
  const labels = language === 'zh'
    ? {system: '系统内置字体：', asset: '应用自带字体：', other: '其他可识别字体：', all: '可识别字体：'}
    : {system: 'Built-in system fonts: ', asset: 'App-bundled fonts: ', other: 'Other supported fonts: ', all: 'Font families: '};
  const assigned = new Set(), expected = [];
  for (const source of ['system', 'asset']) {
    const names = families.filter(family => Array.isArray(info.font_sources?.[family]) && info.font_sources[family].includes(source));
    if (names.length) expected.push({source, text: labels[source] + names.join(' · ')});
    names.forEach(family => assigned.add(family));
  }
  const remaining = families.filter(family => !assigned.has(family));
  if (remaining.length) expected.push({source: null, text: labels[assigned.size ? 'other' : 'all'] + remaining.join(' · ')});
  assert(JSON.stringify(snapshot.fontSources) === JSON.stringify(expected),
    `Font sources must match declared metadata, including legacy unclassified fonts: ${JSON.stringify(snapshot.fontSources)}`);
  const group = info.font_label_groups?.PingFang;
  const hasGroupedPingFang = families.includes('PingFang') && Array.isArray(group)
    && group.includes('PingFang SC') && group.includes('PingFang TC');
  assert(snapshot.fontLabelNote.hidden === !hasGroupedPingFang, 'PingFang scope note must depend on declared grouped labels');
  if (hasGroupedPingFang) {
    const phrase = language === 'zh' ? '苹方覆盖简体／繁体，未细分地区版本' : 'regional variants are not classified separately';
    assert(snapshot.fontLabelNote.text.includes(phrase), `Missing localized grouped PingFang scope: ${JSON.stringify(snapshot.fontLabelNote)}`);
  }
}

async function stopProcess(child) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  let exited = false;
  const exit = new Promise(resolve => child.once('exit', () => { exited = true; resolve(); }));
  child.kill('SIGTERM');
  await Promise.race([exit, sleep(5000)]);
  if (!exited) { child.kill('SIGKILL'); await Promise.race([exit, sleep(2000)]); }
}

async function observeQueue(command, job, first) {
  const timeline = [first];
  const deadline = Date.now() + 180000;
  while (Date.now() < deadline) {
    const item = await evaluate(command, `(() => {
      const state = window.FluxGlyphUI.get();
      if (state.job !== ${JSON.stringify(job)} || !state.snapshot) return null;
      return {status: state.snapshot.status, position: state.queue?.position, ahead: state.queue?.ahead, total: state.queue?.total, progress: state.progress};
    })()`);
    if (item) {
      const previous = timeline[timeline.length - 1];
      if (!previous || JSON.stringify(item) !== JSON.stringify(previous)) timeline.push(item);
      const queuedSoFar = timeline.filter(entry => entry.status === 'queued' && Number.isInteger(entry.position));
      if (queuedSoFar.length >= 2 && queuedSoFar.some(entry => entry.position < queuedSoFar[0].position)) break;
      if (item.status !== 'queued') break;
    }
    await sleep(150);
  }
  const queued = timeline.filter(item => item.status === 'queued' && Number.isInteger(item.position));
  assert(queued.length >= 2, `Did not observe multiple real queue snapshots: ${JSON.stringify(timeline)}`);
  assert(queued.some(item => item.position < queued[0].position), `Queue position never moved down: ${JSON.stringify(timeline)}`);
  for (let index = 1; index < queued.length; index += 1) {
    assert(queued[index].position <= queued[index - 1].position, `Queue position moved backward: ${JSON.stringify(timeline)}`);
  }
  for (const item of timeline) {
    if (item.progress) {
      assert(PROGRESS_STAGES.has(item.progress.stage_code), `Unknown progress stage: ${JSON.stringify(item.progress)}`);
      assert(item.status === 'complete' || item.progress.percent !== 100, `Incomplete job falsely showed 100%: ${JSON.stringify(item)}`);
    }
  }
  return timeline;
}

async function runCase(testCase, activeVersion) {
  const profileDirectory = fs.mkdtempSync(path.join(os.tmpdir(), `flux-glyph-${testCase.name}-`));
  const downloadDirectory = fs.mkdtempSync(path.join(os.tmpdir(), `flux-glyph-download-${testCase.name}-`));
  const badFixture = path.join(os.tmpdir(), `flux-glyph-invalid-${process.pid}-${testCase.name}.png`);
  fs.writeFileSync(badFixture, Buffer.from('not an image'));
  const chrome = spawn(CHROME, ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0', `--user-data-dir=${profileDirectory}`, BASE_URL], {stdio: ['ignore', 'ignore', 'pipe']});
  let chromeError = '';
  chrome.stderr.on('data', chunk => { chromeError += String(chunk); });

  try {
    const port = await waitForDebuggingPort(profileDirectory);
    const page = await waitForPage(port);
    const {socket, opened, command, diagnostics} = connect(page.webSocketDebuggerUrl);
    await opened;
    try {
      await Promise.all([command('Page.enable'), command('Runtime.enable'), command('Log.enable'), command('DOM.enable'), command('Network.enable')]);
      await command('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath: downloadDirectory, eventsEnabled: true});
      await command('Browser.grantPermissions', {origin: BASE_URL, permissions: ['clipboardReadWrite', 'clipboardSanitizedWrite']});
      await command('Emulation.setDeviceMetricsOverride', {width: testCase.width, height: testCase.height, deviceScaleFactor: 1, mobile: testCase.mobile});
      await command('Network.emulateNetworkConditions', {offline: false, latency: 100, downloadThroughput: 5 * 1024 * 1024, uploadThroughput: 5 * 1024 * 1024, connectionType: 'wifi'});
      await command('Page.reload', {ignoreCache: true});
      await waitFor(command, "document.readyState === 'complete' && !!window.FluxGlyphUI", 'application load');
      await waitFor(command, "document.getElementById('download-model')?.getAttribute('aria-disabled') === 'false'", 'available font model download card');
      const fontModel = await evaluate(command, `(async () => {
        const response=await fetch('/api/models/font'), info=await response.json(), link=document.getElementById('download-model');
        return {status:response.status,info,href:link.getAttribute('href'),enabled:link.getAttribute('aria-disabled')==='false',
          visibleVersion:document.getElementById('model-info').textContent,usage:document.getElementById('model-usage-command').textContent};
      })()`, true);
      assert(fontModel.status === 200 && fontModel.info.available && fontModel.info.version === activeVersion &&
        fontModel.enabled && fontModel.href === '/api/models/font/download' && fontModel.visibleVersion.includes(activeVersion) &&
        Array.isArray(fontModel.info.families) && fontModel.info.families.length > 0 &&
        fontModel.info.ocr_required === false && fontModel.usage.includes('python predict.py text-region.png'),
        `Font model download card differs from the active region model: ${JSON.stringify(fontModel)}`);

      const branding = await evaluate(command, `(async () => {
        const heading=document.querySelector('.page-heading h1'), icon=document.querySelector('link[rel="icon"]');
        const response=await fetch(icon.href), source=await response.text();
        return {heading:heading.textContent,fontSize:getComputedStyle(heading).fontSize,footerAbsent:!document.querySelector('footer'),
          favicon:{href:icon.getAttribute('href'),status:response.status,type:response.headers.get('content-type'),red:/#DC2626/i.test(source),white:/#FFF/i.test(source),viewBox:/viewBox="0 0 64 64"/.test(source)}};
      })()`, true);
      const expectedHeadingSize = testCase.mobile ? '32px' : '40px';
      assert(branding.heading === 'Flux Glyph' && branding.fontSize === expectedHeadingSize && branding.footerAbsent, `Flux Glyph branding is incorrect: ${JSON.stringify(branding)}`);
      assert(branding.favicon.status === 200 && /svg/i.test(branding.favicon.type) && branding.favicon.red && branding.favicon.white && branding.favicon.viewBox,
        `Red/white fg favicon is incorrect: ${JSON.stringify(branding.favicon)}`);

      const initial = await switchLanguage(command, 'zh');
      assert(initial.value === 'zh' && initial.lang === 'zh-CN' && initial.pressed.zh === 'true' && initial.pressed.en === 'false', 'Chinese language selection was not applied');
      assert(initial.uploadTitle === '上传图片' && initial.start === '开始识别' && initial.copyJson === '复制 JSON', `Unexpected Chinese controls: ${JSON.stringify(initial)}`);
      assertFontSourceLabels(fontModel.info, initial, 'zh');
      await evaluate(command, "document.querySelector('#model-usage summary').click(); window.scrollTo(0,0); true");
      fontModel.layout = await evaluate(command, `(() => ({
        expanded:document.getElementById('model-usage').open,
        precedesUpload:document.getElementById('font-model').getBoundingClientRect().bottom<=document.querySelector('.upload-help').getBoundingClientRect().top,
        noOverflow:document.documentElement.scrollWidth<=innerWidth+1,
        usageVisible:document.getElementById('model-usage-command').getBoundingClientRect().height>0
      }))()`);
      assert(fontModel.layout.expanded && fontModel.layout.precedesUpload && fontModel.layout.noOverflow && fontModel.layout.usageVisible,
        `Top model card and expanded usage must fit the viewport: ${JSON.stringify(fontModel.layout)}`);
      const modelScreenshot = await captureScreenshot(command, `browser_${testCase.name}_model_sources.png`);
      await evaluate(command, "document.querySelector('#model-usage summary').click(); true");

      const documentNode = await command('DOM.getDocument');
      const fileNode = await command('DOM.querySelector', {nodeId: documentNode.root.nodeId, selector: '#file'});
      assert(fileNode.nodeId, 'File input was not found');
      await command('DOM.setFileInputFiles', {nodeId: fileNode.nodeId, files: [FIXTURE]});
      await waitFor(command, "document.getElementById('filename').textContent === 'ui_title_billing_details.png' && !document.getElementById('start').disabled", 'file selection');

      const queueFillers = testCase.forceQueue ? await submitQueueFillers(8) : [];
      await evaluate(command, "document.getElementById('start').click(); true");
      let processing = null;
      let queueTimeline = [];
      let reconnect = null;
      if (testCase.forceQueue) {
        processing = await waitFor(command, `(() => {
          const state = window.FluxGlyphUI.get(), panel = document.getElementById('progress-panel'), queue = document.getElementById('queue-status');
          if (!state.snapshot || state.snapshot.status !== 'queued' || !state.progress || panel.hidden || queue.hidden) return null;
          const nodes = [panel, document.getElementById('progress-track'), document.getElementById('progress-bar')];
          return {job: state.job, status: state.snapshot.status, progress: state.progress, queue: state.queue,
            label: document.getElementById('progress-label').textContent, queueText: queue.textContent,
            animated: nodes.map(getComputedStyle).some(style => style.animationName !== 'none' || style.transitionDuration.split(',').some(value => parseFloat(value) > 0)),
            visible: panel.getBoundingClientRect().width > 0 && panel.getBoundingClientRect().height > 0};
        })()`, 'visible real queued progress', 20000, 40);
        assert(processing.progress.stage_code === 'queued' && (processing.progress.percent === null || processing.progress.percent === 0), `Queued progress is invalid: ${JSON.stringify(processing)}`);
        assert(processing.visible && processing.animated, `Queued progress is not visibly animated: ${JSON.stringify(processing)}`);
        assert(Number.isInteger(processing.queue.position) && processing.queue.position > 0 && Number.isInteger(processing.queue.ahead), `Queue counts are invalid: ${JSON.stringify(processing.queue)}`);
        assert(Number.isInteger(processing.queue.total) && processing.queue.total >= processing.queue.position, `Queue total is invalid: ${JSON.stringify(processing.queue)}`);
        assert(/排队|等待/.test(`${processing.label} ${processing.queueText}`), `Chinese queue copy is missing: ${JSON.stringify(processing)}`);
        processing.screenshot = await captureScreenshot(command, 'browser_processing.png');

        const pendingLanguages = await evaluate(command, `(() => {
          const control = document.getElementById('language');
          control.querySelector('[data-language="en"]').click();
          const en = {label: document.getElementById('progress-label').textContent, queue: document.getElementById('queue-status').textContent, status: document.getElementById('status').textContent};
          control.querySelector('[data-language="zh"]').click();
          const zh = {label: document.getElementById('progress-label').textContent, queue: document.getElementById('queue-status').textContent, status: document.getElementById('status').textContent};
          return {en, zh, snapshot: window.FluxGlyphUI.get().snapshot};
        })()`);
        assert(pendingLanguages.snapshot.id === processing.job && pendingLanguages.snapshot.status === 'queued', 'Language switch lost the pending real job snapshot');
        assert(/queue|waiting|starting|job/i.test(`${pendingLanguages.en.label} ${pendingLanguages.en.queue} ${pendingLanguages.en.status}`), `English queue copy is missing: ${JSON.stringify(pendingLanguages.en)}`);
        assert(/排队|等待/.test(`${pendingLanguages.zh.label} ${pendingLanguages.zh.queue} ${pendingLanguages.zh.status}`), `Chinese queue copy is missing: ${JSON.stringify(pendingLanguages.zh)}`);
        processing.languages = pendingLanguages;

        await command('Emulation.setEmulatedMedia', {features: [{name: 'prefers-reduced-motion', value: 'reduce'}]});
        const reducedMotion = await evaluate(command, `(() => {
          const bar = getComputedStyle(document.getElementById('progress-bar'));
          return {animationName: bar.animationName, animationDuration: bar.animationDuration, transitionDuration: bar.transitionDuration};
        })()`);
        assert(reducedMotion.animationName === 'none' || reducedMotion.animationDuration.split(',').every(value => parseFloat(value) === 0), `Reduced-motion animation remains active: ${JSON.stringify(reducedMotion)}`);
        assert(reducedMotion.transitionDuration.split(',').every(value => parseFloat(value) === 0), `Reduced-motion transition remains active: ${JSON.stringify(reducedMotion)}`);
        processing.reducedMotion = reducedMotion;
        await command('Emulation.setEmulatedMedia', {features: [{name: 'prefers-reduced-motion', value: 'no-preference'}]});
        queueTimeline = await observeQueue(command, processing.job, {status: processing.status, position: processing.queue.position, ahead: processing.queue.ahead, total: processing.queue.total, progress: processing.progress});

        await command('Network.setBlockedURLs', {urls: [`*://*/api/jobs/${processing.job}*`]});
        reconnect = await waitFor(command, `(() => {
          const state=window.FluxGlyphUI.get(), status=document.getElementById('status').textContent;
          return state.job===${JSON.stringify(processing.job)} && /连接暂时中断/.test(status) && {job:state.job,status,startDisabled:document.getElementById('start').disabled,result:state.result};
        })()`, 'transient poll failure with retained job', 15000, 100);
        assert(reconnect.startDisabled && !reconnect.result, `Transient polling failure exposed restart or stale output: ${JSON.stringify(reconnect)}`);
        await command('Network.setBlockedURLs', {urls: []});
      }

      const completion = await waitFor(command, `(() => {
        const state = window.FluxGlyphUI.get();
        if (!document.getElementById('error').hidden) throw new Error(document.getElementById('error').textContent);
        return state.result && {job: state.job, status: document.getElementById('status').textContent, progress: state.progress};
      })()`, 'real asynchronous recognition result');
      assert(completion.progress?.stage_code === 'complete' && completion.progress.percent === 100, `Completion progress is not 100%: ${JSON.stringify(completion.progress)}`);
      await waitFor(command, "document.getElementById('original').complete && document.getElementById('original').naturalWidth > 0 && document.querySelector('#overlay polygon')", 'image and overlay render');

      const result = await evaluate(command, `(async () => {
        const state = window.FluxGlyphUI.get(), result = state.result, original = document.getElementById('original'), overlay = document.getElementById('overlay');
        overlay.querySelector('polygon').dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
        const crop = document.querySelector('#detail img.detail-image'); if (crop) await crop.decode();
        const response = await fetch(result.annotated_image_url); if (!response.ok) throw new Error('Annotated image request failed');
        const blob = await response.blob(), annotated = new Image(), url = URL.createObjectURL(blob); annotated.src = url; await annotated.decode();
        const canvas = document.createElement('canvas'); canvas.width = result.width; canvas.height = result.height;
        const context = canvas.getContext('2d', {willReadFrequently: true}); context.drawImage(original, 0, 0);
        const before = context.getImageData(0, 0, result.width, result.height).data; context.clearRect(0, 0, result.width, result.height); context.drawImage(annotated, 0, 0);
        const after = context.getImageData(0, 0, result.width, result.height).data; let changedPixels = 0;
        for (let i = 0; i < before.length; i += 4) if (before[i] !== after[i] || before[i+1] !== after[i+1] || before[i+2] !== after[i+2] || before[i+3] !== after[i+3]) changedPixels++;
        URL.revokeObjectURL(url);
        const imageBounds = original.getBoundingClientRect(), overlayBounds = overlay.getBoundingClientRect(), layerBounds = document.querySelector('.image-layer').getBoundingClientRect();
        const visualBounds = document.querySelector('.result-visual').getBoundingClientRect(), jsonBounds = document.querySelector('.json-panel').getBoundingClientRect();
        return {job: state.job, regions: result.regions.length, polygons: overlay.querySelectorAll('polygon').length, selected: window.FluxGlyphUI.get().selected,
          modelVersion:result.model_version,fontMethod:result.font_method,ocrPerformed:result.ocr_performed,
          regionMethods:result.regions.map(region=>region.font?.method),
          fontNames:[...new Set(result.regions.flatMap(region=>[region.font?.family,...(region.font?.candidates||[]).map(candidate=>candidate.family)]).filter(Boolean))],
          textStyle:{source:result.regions.find(region=>region.id===window.FluxGlyphUI.get().selected)?.text_style,
            size:document.querySelector('.text-style-detail .text-size')?.textContent||'',
            color:document.querySelector('.text-style-detail .text-color')?.textContent||'',
            swatch:document.querySelector('.text-style-detail .color-swatch')?.style.backgroundColor||''},
          noTranscription:result.regions.every(region=>(region.text==null||region.text==='')&&region.ocr_confidence==null&&!(region.glyphs||[]).length),
          detailText: document.getElementById('detail').innerText, rawGlyphCode: /ctc_/i.test(document.getElementById('detail').innerText),
          crop: crop ? {width: crop.naturalWidth, height: crop.naturalHeight} : null,
          original: {width: original.naturalWidth, height: original.naturalHeight}, annotated: {width: annotated.naturalWidth, height: annotated.naturalHeight},
          changedPixels, declared: {width: result.width, height: result.height}, viewBox: overlay.getAttribute('viewBox'),
          pointsInside: [...overlay.querySelectorAll('polygon')].every(item => [...item.points].every(point => point.x >= 0 && point.y >= 0 && point.x <= result.width && point.y <= result.height)),
          aligned: Math.abs(imageBounds.left-overlayBounds.left)<.5 && Math.abs(imageBounds.top-overlayBounds.top)<.5 && Math.abs(imageBounds.width-overlayBounds.width)<.5 && Math.abs(imageBounds.height-overlayBounds.height)<.5 && Math.abs(layerBounds.width-imageBounds.width)<.5,
          scrollWidth: document.documentElement.scrollWidth, innerWidth, authHidden: document.getElementById('auth-panel').hidden, summary: document.getElementById('summary').textContent,
          viewerFits: document.querySelector('.viewer').scrollWidth<=document.querySelector('.viewer').clientWidth+1,
          regionCropsFit: [...document.querySelectorAll('.region-thumbnail')].every(item => item.scrollWidth <= item.clientWidth+1),
          headingFits: !document.querySelector('.dist p') || getComputedStyle(document.querySelector('.dist p')).gridColumnEnd === '-1',
          jsonText: document.getElementById('json-output').textContent, jsonExact: document.getElementById('json-output').textContent === JSON.stringify(result,null,2),
          ocr: result.regions.map(region => region.text), noDownloadJson: !document.getElementById('download-json'),
          layout: {sideBySide: jsonBounds.left >= visualBounds.right-1 && jsonBounds.top < visualBounds.bottom, stacked: jsonBounds.top >= visualBounds.bottom-1}};
      })()`, true);

      assert(result.regions > 0 && result.regions === result.polygons && result.selected, `Region selection failed: ${JSON.stringify(result)}`);
      assert(result.modelVersion === activeVersion && result.fontMethod === 'region_neural_network' &&
        result.ocrPerformed === false && result.noTranscription, 'Real inference must use the current region model without OCR');
      assert(result.regionMethods.every(method => method === 'region_neural_network') &&
        result.fontNames.every(family => fontModel.info.families.includes(family)),
        'Region font methods and candidate names must match the current downloaded model labels');
      const style = result.textStyle;
      assert(style.source && style.size.startsWith('估计字号') && style.color.startsWith('文字颜色'), 'Selected region must show its size and color labels');
      const px = style.source.font_size_px_estimate;
      assert(style.size.includes(Number.isFinite(px) && px > 0 ? `≈ ${px.toFixed(1).replace(/\.0$/, '')} px` : '待确认'),
        'Visible pixel size must match the source result, including uncertainty');
      const hex = style.source.text_color_hex;
      if (typeof hex === 'string' && /^#[0-9a-f]{6}$/i.test(hex)) {
        const rgb = [1, 3, 5].map(index => parseInt(hex.slice(index, index + 2), 16));
        assert(style.color.includes(hex.toUpperCase()) && style.swatch === `rgb(${rgb.join(', ')})`, 'Visible color and swatch must match the source result');
      } else assert(style.color.includes('待确认') && !style.swatch, 'Missing color must remain uncertain');
      assert(result.detailText.length > 10 && !result.rawGlyphCode, 'Localized region font detail is incomplete');
      assert(await evaluate(command, "!!document.querySelector('.detail-preview .detail-image') && /^R[0-9]+$/.test(document.querySelector('.detail-preview h2').textContent) && !document.querySelector('.detail-glyphs,#ocr-output')"), 'Region detail must show its crop and ID without OCR or glyph panels');
      assert(result.crop?.width > 0 && result.crop?.height > 0, 'Selected crop did not render');
      assert(result.original.width === result.declared.width && result.original.height === result.declared.height, 'Source dimensions do not match result coordinates');
      assert(result.annotated.width === result.original.width && result.annotated.height === result.original.height && result.changedPixels > 0, 'Annotated image dimensions or pixels are invalid');
      assert(result.viewBox === `0 0 ${result.original.width} ${result.original.height}` && result.pointsInside && result.aligned, 'SVG source-coordinate alignment failed');
      assert(result.scrollWidth <= result.innerWidth + 1 && result.viewerFits && result.regionCropsFit && result.headingFits, `Responsive content is clipped: ${JSON.stringify(result)}`);
      assert(result.authHidden && result.jsonExact && result.noDownloadJson, 'Auth, JSON formatting, or legacy JSON-download contract failed');
      assert(result.layout.stacked, `JSON must follow the visual/detail sections at every width: ${JSON.stringify(result.layout)}`);

      // Force the insecure-context fallback path, click the real control, then restore clipboard access and inspect its exact contents.
      await evaluate(command, "Object.defineProperty(navigator,'clipboard',{configurable:true,value:undefined}); document.getElementById('copy-json').click(); true", false, true);
      const copyZh = await waitFor(command, "document.getElementById('copy-status').textContent === 'JSON 已复制' && document.getElementById('copy-status').textContent", 'fallback copy confirmation');
      await evaluate(command, "delete navigator.clipboard; true");
      const fallbackClipboard = await evaluate(command, 'navigator.clipboard.readText()', true);
      assert(fallbackClipboard === result.jsonText, 'Fallback clipboard copy did not preserve the full exact JSON');

      const zh = await switchLanguage(command, 'zh');
      assert(zh.resultTitle === '识别结果' && zh.copyJson === '复制 JSON' && zh.status === '识别完成' && zh.progressLabel === '识别完成', `Chinese final UI is not localized: ${JSON.stringify(zh)}`);
      const screenshots = {modelSources: modelScreenshot};
      if (!testCase.mobile) screenshots.desktopZh = await captureScreenshot(command, 'browser_desktop_zh.png');
      const en = await switchLanguage(command, 'en');
      assert(en.value === 'en' && en.pressed.en === 'true' && en.pressed.zh === 'false', `English language state is incorrect: ${JSON.stringify(en)}`);
      assert(en.uploadTitle === 'Upload image' && en.start === 'Start recognition' && en.resultTitle === 'Recognition results' && en.copyJson === 'Copy JSON' && en.downloadPng === 'Download annotated PNG', `English controls are not localized: ${JSON.stringify(en)}`);
      assert(en.status === 'Recognition complete' && en.progressLabel === 'Recognition complete', `English progress is not localized: ${JSON.stringify(en)}`);
      assertFontSourceLabels(fontModel.info, en, 'en');
      assert(en.fontReason && !/[\u3400-\u9fff]/.test(en.fontReason) && en.regionCrop && /^R[0-9]+$/.test(en.regionTitle), `English region detail is incomplete: ${JSON.stringify(en)}`);
      assert(en.json === result.jsonText && JSON.stringify(en.ocr) === JSON.stringify(result.ocr), 'Language switching changed raw JSON or original text fields');
      await evaluate(command, "document.getElementById('copy-json').click(); true", false, true);
      const copyEn = await waitFor(command, "document.getElementById('copy-status').textContent === 'JSON copied' && document.getElementById('copy-status').textContent", 'English copy confirmation');
      const clipboard = await evaluate(command, 'navigator.clipboard.readText()', true);
      assert(clipboard === result.jsonText, 'Clipboard API copy did not preserve the full exact JSON');
      if (!testCase.mobile) screenshots.desktopEn = await captureScreenshot(command, 'browser_desktop_en.png');
      await switchLanguage(command, 'zh');
      if (testCase.mobile) screenshots.mobile = await captureScreenshot(command, 'browser_mobile.png');

      const priorDownloads = new Set(fs.readdirSync(downloadDirectory));
      await evaluate(command, "document.getElementById('download-png').click(); true");
      const pngPath = await waitForDownload(downloadDirectory, priorDownloads, 'annotated PNG download');
      const downloadedPng = fs.readFileSync(pngPath), dimensions = pngDimensions(downloadedPng);
      assert(dimensions.width === result.original.width && dimensions.height === result.original.height, 'Downloaded annotated PNG changed dimensions');

      // A new file and a real rejected upload must clear prior results and never show false completion.
      await command('DOM.setFileInputFiles', {nodeId: fileNode.nodeId, files: [badFixture]});
      const cleared = await waitFor(command, `(() => { const s=window.FluxGlyphUI.get(); return !s.result && document.getElementById('result-section').hidden && document.getElementById('json-output').textContent==='' && document.getElementById('copy-json').disabled && document.getElementById('download-png').getAttribute('aria-disabled')==='true'; })()`, 'stale output clearing');
      assert(cleared, 'New file did not clear stale result controls');
      await evaluate(command, "document.getElementById('start').click(); true");
      const errorState = await waitFor(command, `(() => {
        if (document.getElementById('error').hidden) return null;
        const s=window.FluxGlyphUI.get(), track=document.getElementById('progress-track');
        return {message:document.getElementById('error').textContent,result:s.result,json:document.getElementById('json-output').textContent,
          resultHidden:document.getElementById('result-section').hidden,copyDisabled:document.getElementById('copy-json').disabled,
          progressInvalid:track.getAttribute('aria-invalid'),progressNow:track.getAttribute('aria-valuenow')};
      })()`, 'real invalid-upload error');
      assert(!errorState.result && !errorState.json && errorState.resultHidden && errorState.copyDisabled, `Error state exposed stale output: ${JSON.stringify(errorState)}`);
      assert(errorState.progressInvalid === 'true' && errorState.progressNow !== '100', `Error state showed false completion: ${JSON.stringify(errorState)}`);

      let docsLocalization = null;
      let faviconScreenshot = null;
      if (!testCase.mobile) {
        await switchLanguage(command, 'en');
        await command('Page.navigate', {url: `${BASE_URL}/docs`});
        docsLocalization = await waitFor(command, `(() => {
          const heading=document.querySelector('h2[data-lang="en"]:not([hidden])');
          const brand=document.querySelector('.page-heading h1');
          return document.readyState==='complete' && document.documentElement.lang==='en' && document.getElementById('language')?.dataset.language==='en' && heading && {lang:document.documentElement.lang,heading:heading.textContent,brand:brand.textContent,fontSize:getComputedStyle(brand).fontSize,footerAbsent:!document.querySelector('footer')};
        })()`, 'English API documentation inherited language preference');
        assert(docsLocalization.heading === 'Download and run the font model' && docsLocalization.brand === 'Flux Glyph API' && docsLocalization.fontSize === '32px' && docsLocalization.footerAbsent,
          `API documentation branding or language is incorrect: ${JSON.stringify(docsLocalization)}`);
        await command('Emulation.setDeviceMetricsOverride', {width: 64, height: 64, deviceScaleFactor: 1, mobile: false});
        await command('Page.navigate', {url: `${BASE_URL}${branding.favicon.href}`});
        await waitFor(command, "document.readyState === 'complete'", 'favicon document');
        faviconScreenshot = await captureScreenshot(command, 'browser_favicon.png');
      }

      const expectedInvalidUploadDiagnostic = message => message.includes(`${BASE_URL}/api/jobs`) &&
        (message.includes('http 400:') || message.includes('status of 400'));
      const applicationErrors = diagnostics.filter(message => !message.includes('/favicon.ico') && !expectedInvalidUploadDiagnostic(message));
      assert(applicationErrors.length === 0, `Browser errors: ${applicationErrors.join(' | ')}`);
      return {
        viewport: {width: testCase.width, height: testCase.height, mobile: testCase.mobile}, job: result.job, status: completion.status,
        fontModel,modelVersion:result.modelVersion,fontMethod:result.fontMethod,ocrPerformed:result.ocrPerformed,noTranscription:result.noTranscription,
        progress: completion.progress, processing: processing ? {...processing, queueFillers: queueFillers.length, timeline: queueTimeline, reconnect} : null,
        regions: result.regions, original: result.original, annotated: result.annotated, changedPixels: result.changedPixels, crop: result.crop, viewBox: result.viewBox,
        noHorizontalOverflow: true, viewerFitsImage: result.viewerFits, regionCropThumbnailsFit: result.regionCropsFit, candidateHeadingSpansColumns: result.headingFits,
        selectedRegion: result.selected, summary: result.summary, layout: result.layout, textStyle: result.textStyle,
        json: {formattedExact: result.jsonExact, bytes: Buffer.byteLength(result.jsonText), legacyDownloadAbsent: result.noDownloadJson},
        clipboard: {fallback: copyZh, api: copyEn, exact: clipboard === fallbackClipboard && clipboard === result.jsonText},
        localization: {chinese: {uploadTitle: zh.uploadTitle, resultTitle: zh.resultTitle, progress: zh.progressLabel, fontReason: zh.fontReason, fontSources: zh.fontSources, fontLabelNote: zh.fontLabelNote}, english: {uploadTitle: en.uploadTitle, resultTitle: en.resultTitle, progress: en.progressLabel, fontReason: en.fontReason, fontSources: en.fontSources, fontLabelNote: en.fontLabelNote}, ocrUnchanged: true, rawJsonUnchanged: true},
        staleOutputCleared: cleared, errorState, branding, docsLocalization, faviconScreenshot, download: {png: {filename: path.basename(pngPath), bytes: downloadedPng.length, ...dimensions}}, screenshots,
        browserErrors: applicationErrors, ignoredBrowserDiagnostics: diagnostics.filter(message => message.includes('/favicon.ico') || expectedInvalidUploadDiagnostic(message)),
      };
    } finally { socket.close(); }
  } catch (error) {
    if (chromeError) error.message += `\nChrome stderr:\n${chromeError.slice(-2000)}`;
    throw error;
  } finally {
    await stopProcess(chrome);
    for (const target of [profileDirectory, downloadDirectory]) fs.rmSync(target, {recursive: true, force: true});
    fs.rmSync(badFixture, {force: true});
  }
}

(async () => {
  for (const required of [CHROME, FIXTURE]) assert(fs.existsSync(required), `Required QA file not found: ${required}`);
  queueSource = createSyntheticQueueFixture();
  fs.mkdirSync(OUTPUT, {recursive: true});
  const health = await waitForServerIdle();
  assert(health.workers === 1, `Expected one worker, got ${health.workers}`);
  assert(health.queue_size >= 8, `Expected at least eight waiting slots, got ${health.queue_size}`);
  const activeVersion = health.model_version;
  assert(typeof activeVersion === 'string' && activeVersion.length > 0, 'Health must identify the active model version');
  if (PINNED_MODEL) assert(activeVersion === PINNED_MODEL, `Expected deployment ${PINNED_MODEL}, got ${activeVersion}`);
  const cases = [];
  for (const testCase of CASES) {
    const current = await waitForServerIdle();
    assert(current.model_version === activeVersion, 'Active model changed during browser QA; rerun against one stable deployment');
    cases.push(await runCase(testCase, activeVersion));
  }
  const report = {
    result: 'passed', testedAt: new Date().toISOString(), baseUrl: BASE_URL, health,
    checks: [
      'real upload, one-worker FIFO queue position downshift, and asynchronous polling',
      'visible animated stage progress without false 100%, plus reduced-motion behavior',
      'Chinese/English switching during pending and completed states',
      'region results followed by full-width exact formatted JSON on desktop and mobile',
      'health-selected active model download card, matching version, same-origin link, and standalone usage',
      'metadata-driven system/app font sources and grouped PingFang scope in Chinese/English; legacy metadata remains compatible',
      'real region inference returns no OCR text or single-character evidence',
      'exact full JSON copy through Clipboard API and insecure-context fallback',
      'no legacy JSON download control; original-size annotated PNG still downloads',
      'polygon selection, crop, font evidence, source coordinates, and pixel annotation',
      'region pixel size and text color/swatch match the source result without inventing uncertain values',
      'Original result fields and raw JSON unchanged by localization',
      'new selection and real upload error clear stale result/JSON controls',
      'desktop/mobile overflow and browser diagnostics',
      'responsive 40/32px Flux Glyph branding, no footer, and a loaded red/white SVG favicon',
    ], cases,
  };
  fs.writeFileSync(path.join(OUTPUT, 'BROWSER_QA.json'), `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report, null, 2));
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; }).finally(() => {
  if (queueSource) fs.rmSync(queueSource, {force: true});
});
