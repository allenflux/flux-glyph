(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const svg = (tag, attributes) => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    return node;
  };

  const messages = {
    zh: {
      fontModelTitle: '字体模型', downloadModel: '下载字体模型', fontModelHelp: '下载后可在本地输入文字区域裁图，识别字体并估计字号、颜色。',
      modelLoading: '正在检查模型下载…', modelUnavailable: '当前暂无可下载的字体模型，可稍后刷新状态。', modelLoadFailed: '暂时无法获取下载信息，请点“刷新状态”重试。', modelLocked: '请先在上方输入访问令牌解锁，再下载模型。',
      refreshModel: '刷新状态', modelUsageToggle: '本地使用方式',
      modelAvailable: '{version} · {size}', modelFamilies: '可识别字体：{families}', modelUsageHelp: '解压下载包，在解压目录安装依赖并运行：', modelUsageDocs: '模型使用说明',
      modelSystemFonts: '系统内置字体：{families}', modelAssetFonts: '应用自带字体：{families}', modelOtherFonts: '其他可识别字体：{families}',
      modelPingFangGroup: '苹方覆盖简体／繁体，未细分地区版本。',
      regionsTitle: '文字区域', fontDetailTitle: '字体详情', closeDetail: '收起详情', cropTitle: '原图区域裁图', fontConclusionTitle: '字体判断',
      tagline: '上传截图，识别字体、估计字号与文字颜色。', language: '语言', apiExample: 'API 示例',
      unlockTitle: '解锁访问', unlockHelp: '此服务需要访问令牌。令牌只在本次页面会话中用于设置安全 Cookie，不会保存在浏览器存储中。', tokenPlaceholder: '输入访问令牌', unlock: '解锁',
      uploadTitle: '上传图片', uploadHelp: '支持 PNG、JPG、WebP。直接分析文字区域的外观，查看字体候选、字号和颜色。', dropPrompt: '选择图片或拖入这里', noFile: '尚未选择图片', start: '开始识别', downloadPng: '下载标注 PNG',
      waiting: '等待上传', selected: '已选择 {name}，等待开始识别', creating: '正在创建任务…', queued: '排队中…', running: '正在识别…', complete: '识别完成', failed: '识别失败', pollFailed: '轮询失败：{message}', reconnecting: '连接暂时中断，{seconds} 秒后继续查询当前任务…', jobGone: '任务已不存在，请再次开始识别。', authRequired: '登录已失效，请重新解锁。', chooseImage: '请选择单张图片',
      preparing: '正在准备图片…', detecting: '正在框选文字区域…', recognizing: '正在分析区域…', classifying: '正在识别字体…', matching: '正在识别字体…', annotating: '正在生成字体标注图片…', finalizing: '正在整理结果…', progressLabel: '阶段进度', progressCount: '{current}/{total}', queuePosition: '排队第 {position} 位，前方 {ahead} 个任务', queueNext: '即将开始处理', busy: '当前排队任务较多，请稍后再试。图片仍保留，可直接再次开始识别。', invalidImage: '图片无法读取。请上传 8 MB 内、1200 万像素内的 PNG、JPG 或 WebP 静态图。', imageTooLarge: '图片超过 8 MB 上传限制，请选择较小的图片。', uploadFailed: '任务创建失败，请检查连接后再次开始识别。', restarted: '服务已重启，请再次开始识别。',
      resultsTitle: '识别结果', detailEmpty: '点击图片中的文字框或区域列表查看字体证据。', noRegions: '未检测到文字区域。请使用清晰、未裁切的截图重试。', originalAlt: '后端已校正方向的原图', overlayLabel: '检测到的文字区域', regionLabel: '文字区域 {id}', cropAlt: '所选文字区域裁图', thumbnailAlt: '{id} 原图区域', cropUnavailable: '暂无裁图',
      copyJson: '复制 JSON', copied: 'JSON 已复制', copyFailed: '无法自动复制，请在 JSON 区域中手动选择复制。', jsonLabel: '格式化 JSON',
      legendLabel: '字体结论图例',
      detected: '检测 {count}', chinese: '中文 {count}', pingfang: '苹方支持 {count}', seconds: '{count} 秒',
      fontCandidates: '字体候选 {count}', scopeLatin: '此结论仅针对框内数字和英文字母；标点、图标不参与字体判断。', scopeMixed: '混排文字按中文与数字／英文分别匹配；上方结论针对中文部分。', chinesePart: '中文部分', latinPart: '数字／英文部分',
      regionScope: '此结果针对当前文字区域的整体外观。', neuralMethod: '神经网络', neuralIdentified: '已识别 {count}', topScores: 'Top 3 模型分数（分数不是实际准确率）',
      closestFont: '最接近：{family}', modelScore: '模型评分 {score}', noModelScore: '未生成评分',
      rejectedFont: '未知字体', colorOnlyNote: '颜色为截图中的可见颜色。',
      estimatedSize: '估计字号', textColor: '文字颜色', styleUnknown: '待确认', styleNote: '字号按截图像素估计；颜色为截图中的可见颜色。', sizeRange: '估计范围 {low}–{high} px',
      supported: '支持', candidate: '候选', uncertain: '待确认', outOfScope: '字体未覆盖', unknownText: '未识别文字', unknownFont: '待确认', noReason: '没有进一步说明', scopeNote: '此结论仅针对框内中文，不能据此推断数字和英文字体。', topDistances: 'Top 3 原始距离（距离不是概率）',
      modelVersion: '模型版本：{version}', healthUnavailable: '服务状态暂不可用', unlocked: '已解锁，可以上传图片', authError: '令牌无效或服务拒绝访问，请重试。',
      apiHelp: '上传支付宝图片，完成后直接返回字体识别 JSON。', apiAsync: '使用 ?wait=false 先返回任务 ID，再轮询 GET /api/jobs/{id}。仍兼容 POST /api/predict?wait=true。', apiDocs: '完整 API 说明'
    },
    en: {
      fontModelTitle: 'Font model', downloadModel: 'Download font model', fontModelHelp: 'Run the model locally on a cropped text region to identify its font and estimate size and color.',
      modelLoading: 'Checking model download…', modelUnavailable: 'No font model is available to download yet. Check again later.', modelLoadFailed: 'Download information could not be loaded. Select Refresh status to try again.', modelLocked: 'Enter your access token above to unlock the model download.',
      refreshModel: 'Refresh status', modelUsageToggle: 'Run locally',
      modelAvailable: '{version} · {size}', modelFamilies: 'Font families: {families}', modelUsageHelp: 'Extract the download, then install dependencies and run these commands in its directory:', modelUsageDocs: 'Model usage guide',
      modelSystemFonts: 'Built-in system fonts: {families}', modelAssetFonts: 'App-bundled fonts: {families}', modelOtherFonts: 'Other supported fonts: {families}',
      modelPingFangGroup: 'PingFang covers Simplified and Traditional Chinese; regional variants are not classified separately.',
      regionsTitle: 'Text regions', fontDetailTitle: 'Font details', closeDetail: 'Hide details', cropTitle: 'Source crop', fontConclusionTitle: 'Font result',
      tagline: 'Upload a screenshot to identify fonts, estimated sizes, and text colors.', language: 'Language', apiExample: 'API examples',
      unlockTitle: 'Unlock access', unlockHelp: 'This service requires an access token. It is used only to set a secure cookie for this page session and is never saved in browser storage.', tokenPlaceholder: 'Enter access token', unlock: 'Unlock',
      uploadTitle: 'Upload image', uploadHelp: 'Supports PNG, JPG, and WebP. Analyze the appearance of each text region to estimate its font, size, and color.', dropPrompt: 'Choose an image or drop it here', noFile: 'No image selected', start: 'Start recognition', downloadPng: 'Download annotated PNG',
      waiting: 'Waiting for an image', selected: '{name} selected. Ready to start.', creating: 'Creating job…', queued: 'Waiting in queue…', running: 'Recognizing…', complete: 'Recognition complete', failed: 'Recognition failed', pollFailed: 'Status check failed: {message}', reconnecting: 'Connection interrupted. Checking this job again in {seconds} seconds…', jobGone: 'This job is no longer available. Start recognition again.', authRequired: 'Your access session expired. Unlock the service again.', chooseImage: 'Please choose one image',
      preparing: 'Preparing image…', detecting: 'Detecting text regions…', recognizing: 'Analyzing regions…', classifying: 'Identifying fonts…', matching: 'Identifying fonts…', annotating: 'Generating annotated image…', finalizing: 'Finalizing results…', progressLabel: 'Stage progress', progressCount: '{current}/{total}', queuePosition: 'Queue position {position} · {ahead} job(s) ahead', queueNext: 'Starting soon', busy: 'The queue is currently full. Try again shortly; your selected image is still ready.', invalidImage: 'The image could not be read. Upload a static PNG, JPG, or WebP under 8 MB and 12 megapixels.', imageTooLarge: 'The image exceeds the 8 MB upload limit. Choose a smaller image.', uploadFailed: 'The job could not be created. Check your connection and start recognition again.', restarted: 'The service restarted before this job finished. Start recognition again.',
      resultsTitle: 'Recognition results', detailEmpty: 'Select a box in the image or a region in the list to inspect its font evidence.', noRegions: 'No text regions were found. Try a clear, uncropped screenshot.', originalAlt: 'Source image with orientation corrected by the server', overlayLabel: 'Detected text regions', regionLabel: 'Text region {id}', cropAlt: 'Crop of the selected text region', thumbnailAlt: 'Source region {id}', cropUnavailable: 'Crop unavailable',
      copyJson: 'Copy JSON', copied: 'JSON copied', copyFailed: 'Automatic copy failed. Select the formatted JSON and copy it manually.', jsonLabel: 'Formatted JSON',
      legendLabel: 'Font conclusion legend',
      detected: 'Detected {count}', chinese: 'Chinese {count}', pingfang: 'PingFang supported {count}', seconds: '{count} sec',
      fontCandidates: 'Font candidates {count}', scopeLatin: 'This verdict covers digits and Latin letters only; punctuation and icons are not classified.', scopeMixed: 'Chinese and numeric/Latin text are matched separately. The main verdict covers Chinese glyphs.', chinesePart: 'Chinese text', latinPart: 'Numeric / Latin text',
      regionScope: 'This result describes the appearance of the selected text region.', neuralMethod: 'Neural network', neuralIdentified: 'Identified {count}', topScores: 'Top 3 model scores (scores are not measured accuracy)',
      closestFont: 'Closest match: {family}', modelScore: 'Model score {score}', noModelScore: 'No model score',
      rejectedFont: 'Unknown font', colorOnlyNote: 'Color is the visible color in the screenshot.',
      estimatedSize: 'Estimated size', textColor: 'Text color', styleUnknown: 'Uncertain', styleNote: 'Size is estimated in screenshot pixels; color is the visible color in the screenshot.', sizeRange: 'Estimated range {low}–{high} px',
      supported: 'Supported', candidate: 'Candidate', uncertain: 'Review', outOfScope: 'Font out of scope', unknownText: 'Unrecognized text', unknownFont: 'Review needed', noReason: 'No further explanation is available.', scopeNote: 'This verdict covers Chinese glyphs only and does not determine numeric or Latin fonts.', topDistances: 'Top 3 raw distances (distance is not probability)',
      modelVersion: 'Model version: {version}', healthUnavailable: 'Service status is unavailable', unlocked: 'Unlocked. You can upload an image.', authError: 'The token is invalid or the service refused access. Try again.',
      apiHelp: 'Upload a payment screenshot and receive font-recognition JSON when processing completes.', apiAsync: 'Use ?wait=false to receive a job ID first, then poll GET /api/jobs/{id}. POST /api/predict?wait=true remains supported.', apiDocs: 'Full API documentation'
    }
  };

  let language = loadLanguage();
  let file = null;
  let job = null;
  let timer = null;
  let result = null;
  let selected = null;
  let preview = null;
  let generation = 0;
  let statusState = {key: 'waiting', values: {}};
  let progressState = null;
  let latestSnapshot = null;
  let pollFailures = 0;
  let reconnectState = null;
  let modelInfo = null;
  let modelLoadState = 'modelLoading';
  let modelLoading = false;

  function loadLanguage() {
    try { return localStorage.getItem('flux-glyph-language') === 'en' ? 'en' : 'zh'; } catch (_) { return 'zh'; }
  }

  function t(key, values = {}) {
    const template = messages[language][key] || messages.zh[key] || key;
    return template.replace(/\{(\w+)\}/g, (_, name) => values[name] == null ? '' : String(values[name]));
  }

  function state(key, error = false, values = {}) {
    statusState = {key, values};
    const text = t(key, values);
    $('status').textContent = text;
    $('error').hidden = !error;
    $('error').textContent = error ? text : '';
  }

  function stop() {
    if (timer) clearTimeout(timer);
    timer = null;
    job = null;
    pollFailures = 0;
    reconnectState = null;
  }

  const api = (url, options = {}) => fetch(url, options).then(async response => {
    let data = {};
    try { data = await response.json(); } catch (_) {}
    if (!response.ok) {
      const error = Error(data.error || data.detail || `${response.status}`);
      error.status = response.status;
      error.data = data;
      throw error;
    }
    return data;
  });

  function resetOutput() {
    result = null;
    selected = null;
    $('result-section').hidden = true;
    $('json-output').textContent = '';
    $('copy-json').disabled = true;
    $('copy-status').textContent = '';
    $('detail-panel').hidden = true;
    for (const id of ['regions', 'detail', 'overlay']) $(id).replaceChildren();
    for (const node of [$('regions'), $('detail'), document.querySelector('.viewer')]) node.scrollTop = 0;
    $('download-png').removeAttribute('href');
    $('download-png').setAttribute('aria-disabled', 'true');
  }

  function resetProgress() {
    progressState = null;
    latestSnapshot = null;
    $('progress-panel').hidden = true;
    $('progress-panel').classList.remove('progress-error');
    $('progress-panel').classList.remove('progress-complete');
    $('progress-panel').removeAttribute('data-stage');
    const track = $('progress-track');
    track.classList.remove('indeterminate');
    track.removeAttribute('data-stage');
    track.removeAttribute('aria-valuenow');
    track.removeAttribute('aria-valuetext');
    track.removeAttribute('aria-invalid');
    $('progress-bar').style.width = '0%';
    $('progress-count').textContent = '';
    $('queue-status').textContent = '';
    $('queue-status').hidden = true;
  }

  function setFile(nextFile) {
    generation++;
    stop();
    file = nextFile;
    resetOutput();
    resetProgress();
    $('start').disabled = !nextFile;
    $('filename').textContent = nextFile ? nextFile.name : t('noFile');
    if (preview) URL.revokeObjectURL(preview);
    preview = nextFile ? URL.createObjectURL(nextFile) : null;
    state(nextFile ? 'selected' : 'waiting', false, nextFile ? {name: nextFile.name} : {});
  }

  function normalizeStageCode(code) {
    return String(code || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  }

  function stageKey(code, raw, status) {
    const value = normalizeStageCode(code);
    if (/queue|pending/.test(value)) return 'queued';
    if (/load|prepare|orient|decode|upload/.test(value)) return 'preparing';
    if (/classif|font|match/.test(value)) return 'matching';
    if (/detect|box|region/.test(value)) return 'detecting';
    if (/ocr|recogn|read|text/.test(value)) return 'recognizing';
    if (/font|match|glyph/.test(value)) return 'matching';
    if (/annotat|render|output|image/.test(value)) return 'annotating';
    if (/final|persist|serializ/.test(value)) return 'finalizing';
    if (/complete|done|success/.test(value)) return 'complete';
    const human = String(raw || '');
    if (/队列|排队/.test(human)) return 'queued';
    if (/准备|加载|校正/.test(human)) return 'preparing';
    if (/框选|检测.*文字|文字区域/.test(human)) return 'detecting';
    if (/识读|识别文字/.test(human)) return 'recognizing';
    if (/匹配字体|识别字体|分析字体/.test(human)) return 'matching';
    if (/标注|生成.*图片/.test(human)) return 'annotating';
    if (/整理|保存.*结果/.test(human)) return 'finalizing';
    if (/完成/.test(human)) return 'complete';
    return status === 'queued' ? 'queued' : status === 'complete' ? 'complete' : 'running';
  }

  function updateProgress(progress, status, rawStage, forceComplete = false) {
    const data = progress && typeof progress === 'object' ? progress : {};
    const code = data.stage_code || '';
    const key = stageKey(code, rawStage, status);
    const numeric = forceComplete ? 100 : (status !== 'queued' && typeof data.percent === 'number' && Number.isFinite(data.percent) ? Math.max(0, Math.min(99, data.percent)) : null);
    const current = Number.isFinite(data.current) ? data.current : null;
    const total = Number.isFinite(data.total) ? data.total : null;
    progressState = {progress: data, status, rawStage, forceComplete};
    const panel = $('progress-panel');
    const track = $('progress-track');
    panel.hidden = false;
    panel.classList.remove('progress-error');
    panel.classList.toggle('progress-complete', forceComplete);
    panel.dataset.stage = forceComplete ? 'complete' : (status === 'queued' ? 'queued' : 'running');
    track.dataset.stage = panel.dataset.stage;
    track.removeAttribute('aria-invalid');
    $('progress-label').textContent = t(key);
    const hasCount = current != null && total != null && total > 0;
    $('progress-count').textContent = hasCount ? t('progressCount', {current, total}) : (numeric == null ? '' : `${Math.round(numeric)}%`);
    track.setAttribute('aria-valuetext', hasCount ? `${t(key)} ${current}/${total}` : (numeric == null ? t(key) : `${t(key)} ${Math.round(numeric)}%`));
    if (numeric == null) {
      track.classList.add('indeterminate');
      track.removeAttribute('aria-valuenow');
      $('progress-bar').style.width = '';
    } else {
      track.classList.remove('indeterminate');
      track.setAttribute('aria-valuenow', String(Math.round(numeric)));
      $('progress-bar').style.width = `${numeric}%`;
    }
    state(key);
  }

  function markProgressError(message) {
    const panel = $('progress-panel');
    const track = $('progress-track');
    panel.hidden = false;
    panel.classList.add('progress-error');
    panel.dataset.stage = 'error';
    track.dataset.stage = 'error';
    panel.classList.remove('progress-complete');
    track.classList.remove('indeterminate');
    track.removeAttribute('aria-valuenow');
    track.setAttribute('aria-invalid', 'true');
    track.setAttribute('aria-valuetext', message);
    $('progress-label').textContent = message;
    $('progress-count').textContent = '';
    $('queue-status').hidden = true;
  }

  function updateQueue(data) {
    const queue = $('queue-status');
    const position = Number.isFinite(data?.queue_position) ? data.queue_position : null;
    const ahead = Number.isFinite(data?.queue_ahead) ? data.queue_ahead : null;
    if (data?.status === 'queued' && position != null) {
      queue.textContent = position <= 1 && (ahead == null || ahead === 0) ? t('queueNext') : t('queuePosition', {position, ahead: ahead ?? Math.max(0, position - 1)});
      queue.hidden = false;
    } else {
      queue.textContent = '';
      queue.hidden = true;
    }
  }

  function start(url = '/api/jobs', options) {
    const activeGeneration = ++generation;
    stop();
    resetOutput();
    resetProgress();
    $('start').disabled = true;
    updateProgress({stage_code: 'preparing', percent: null}, 'running', '', false);
    state('creating');
    api(url, options).then(data => {
      if (activeGeneration !== generation) return;
      job = data.id;
      latestSnapshot = data;
      pollFailures = 0;
      reconnectState = null;
      updateQueue(data);
      updateProgress(data.progress, data.status || 'queued', data.stage);
      poll(activeGeneration);
    }).catch(error => {
      if (activeGeneration !== generation) return;
      $('start').disabled = !file;
      const key = error.status === 429 ? 'busy' : error.status === 413 ? 'imageTooLarge' : (error.status === 400 || error.status === 415) ? 'invalidImage' : (error.status === 401 || error.status === 403) ? 'authRequired' : 'uploadFailed';
      if (error.status === 401 || error.status === 403) $('auth-panel').hidden = false;
      state(key, true);
      markProgressError(t(key));
    });
  }

  function poll(activeGeneration) {
    if (!job || activeGeneration !== generation) return;
    api(`/api/jobs/${encodeURIComponent(job)}`).then(data => {
      if (activeGeneration !== generation) return;
      latestSnapshot = data;
      pollFailures = 0;
      reconnectState = null;
      updateQueue(data);
      if (data.status === 'error') {
        $('start').disabled = !file;
        const errorKey = data.error_code === 'service_restarted' ? 'restarted' : 'failed';
        state(errorKey, true);
        const message = language === 'zh' && data.error_code !== 'service_restarted' && data.error ? data.error : t(errorKey);
        $('status').textContent = message;
        $('error').textContent = message;
        markProgressError(message);
        return;
      }
      updateProgress(data.progress, data.status, data.stage, data.status === 'complete');
      if (data.status === 'complete') {
        $('start').disabled = !file;
        show(data.result);
        return;
      }
      timer = setTimeout(() => poll(activeGeneration), 1000);
    }).catch(error => {
      if (activeGeneration !== generation) return;
      if (error.status === 401 || error.status === 404 || (error.status >= 400 && error.status < 500 && error.status !== 429)) {
        const key = error.status === 401 ? 'authRequired' : error.status === 404 ? 'jobGone' : 'pollFailed';
        const values = key === 'pollFailed' ? {message: error.message} : {};
        job = null;
        timer = null;
        latestSnapshot = null;
        reconnectState = null;
        $('start').disabled = !file;
        if (error.status === 401) $('auth-panel').hidden = false;
        state(key, true, values);
        markProgressError(t(key, values));
        return;
      }
      pollFailures++;
      const delay = Math.min(10000, 2000 * (2 ** Math.min(pollFailures - 1, 3)));
      reconnectState = {seconds: Math.ceil(delay / 1000)};
      state('reconnecting', false, reconnectState);
      $('start').disabled = true;
      timer = setTimeout(() => poll(activeGeneration), delay);
    });
  }

  const font = region => region.font || {label: '', status: 'uncertain', reason: '', candidates: []};
  const points = quad => (quad || []).map(point => point.join(',')).join(' ');
  const isNeural = value => ['neural_network', 'region_neural_network'].includes(value.method || value.font_method);
  const statusKey = value => ({supported: 'supported', candidate: 'candidate', uncertain: 'uncertain', out_of_scope: 'outOfScope'})[value] || 'uncertain';

  function rejectionReason(value) {
    const reasons = ['unknown_font_rejected', 'invalid_rejection_output'];
    if (reasons.includes(value.reason_code)) return value.reason_code;
    if (reasons.includes(value.reason)) return value.reason;
    if (value.rejection?.status === 'rejected') return 'unknown_font_rejected';
    if (value.rejection?.status === 'unavailable') {
      // Preprocessing can stop before the rejection network runs. Preserve
      // that actionable reason while still withholding font names and size.
      const preprocessing = ['low_quality_region', 'nonuniform_region_background', 'region_too_long', 'too_many_regions'];
      return preprocessing.includes(value.reason_code) ? value.reason_code : 'invalid_rejection_output';
    }
    return null;
  }

  function scoredCandidates(value) {
    if (rejectionReason(value) || !isNeural(value) || !Array.isArray(value.candidates)) return [];
    // Sort a filtered copy for display; preserve the server verdict and JSON.
    return value.candidates.filter(candidate => candidate && typeof candidate.family === 'string'
      && candidate.family.trim() && Number.isFinite(candidate.score) && candidate.score >= 0 && candidate.score <= 1)
      .sort((a, b) => b.score - a.score);
  }

  function fontLabel(value) {
    const rejected = rejectionReason(value);
    if (rejected) return t(rejected === 'unknown_font_rejected' ? 'rejectedFont' : 'noModelScore');
    if (value.family) return value.status === 'candidate' ? `${value.family}${language === 'zh' ? '（候选）' : ' (candidate)'}` : value.family;
    return t(statusKey(value.status) === 'outOfScope' ? 'outOfScope' : 'unknownFont');
  }

  const reasonTranslations = {
    '该图文字区域过多；已保留框，本次超出处理上限。': 'This image contains too many text regions. The box is preserved, but this region exceeds the processing limit.',
    '文字未可靠读出，或文字行超出长度限制。': 'The text could not be read reliably, or the line exceeds the length limit.',
    '当前模型识别中文字体，数字和英文保留框。': 'The current model identifies Chinese fonts; Latin letters and numbers retain detection boxes only.',
    '识读文字置信度不足，字体待确认。': 'Text recognition confidence is too low to confirm the font.',
    '字形证据不足。': 'Glyph evidence is insufficient to confirm the font.',
    '字形证据不足或未通过字体确认门槛。': 'Glyph evidence is insufficient or did not pass the font confirmation threshold.',
    '中文字形通过苹方门槛，逐字候选一致。': 'The Chinese glyphs pass the PingFang threshold and agree across character candidates.',
    '该中文字体在三种图像检查中排名一致，字体名仍为候选。': 'This Chinese font ranks consistently across three image checks; the font name remains a candidate.',
    '部分汉字无法可靠分字，保留原图区域供复核。': 'Some Chinese characters could not be segmented reliably; the source region is preserved for review.'
  };

  const reasonCodes = {
    unknown_font_rejected: {zh: '当前模型无法识别该字体，已保留原图区域供查看。', en: 'The model does not recognize this font. The source crop is retained for review.'},
    invalid_rejection_output: {zh: '字体判断结果异常，暂不能确认字体。请重试。', en: 'The font check returned an invalid result. Retry to identify this font.'},
    region_neural_family_candidate: {zh: '区域字体神经网络的模型分数与候选区分度达到当前门槛。', en: 'The region font network passes the current score and separation gates.'},
    low_quality_region: {zh: '区域图像质量不足，字体待确认。', en: 'This region lacks sufficient image quality to identify a font.'},
    nonuniform_region_background: {zh: '区域背景颜色不均匀，字体待确认。', en: 'The region background is not uniform; review is needed.'},
    region_too_long: {zh: '文字区域过长，请裁出较短的文字区域后重试。', en: 'This text region is too long. Crop a shorter region and try again.'},
    mixed_or_ambiguous_region: {zh: '区域内不同图像片段的字体判断不一致，暂未确认。', en: 'Different image patches within the region disagree on the font; review is needed.'},
    neural_family_candidate: {zh: '字体神经网络的模型分数与候选区分度达到当前门槛。', en: 'The trained font network passes the current score and separation gates.'},
    below_score_gate: {zh: '神经网络已有字体候选，但模型分数未达到确认门槛。', en: 'The neural network has candidates, but the model score is below the gate.'},
    ambiguous_neural_families: {zh: '前两名字体的模型评分过于接近，暂未确认。', en: 'The top two font families have similar model scores; review is needed.'},
    incomplete_segmentation: {zh: '部分字形未能可靠分出，神经网络结论待确认。', en: 'Some glyphs could not be segmented reliably; the neural verdict needs review.'},
    low_quality_or_invalid_glyphs: {zh: '部分字形的图像质量不足，神经网络结论待确认。', en: 'Some glyph images lack sufficient quality for a neural verdict.'},
    no_samples: {zh: '没有可用于字体判断的可靠字形。', en: 'No reliable glyphs are available for font classification.'},
    sample_limit_exceeded: {zh: '该区域字数超出本次字体判断上限。', en: 'This region exceeds the font-classification glyph limit.'},
    neural_inference_failed: {zh: '字体神经网络本次推理失败，请重试。', en: 'Font neural inference failed; retry the request.'},
    invalid_neural_output: {zh: '字体神经网络输出异常，本次不作结论。', en: 'Font neural inference returned invalid output; no verdict is made.'},
    unsupported_script: {zh: '当前字库尚未覆盖该文字类型。', en: 'This script is not covered by the current font bank.'},
    stable_latin_family_candidate: {zh: '数字／英文字形通过距离与区分度门槛，且三种图像检查的候选一致。', en: 'Numeric/Latin glyphs pass distance and separation gates and agree across three image checks.'},
    insufficient_latin_evidence: {zh: '数字／英文字形证据不足，字体待确认。', en: 'There is insufficient numeric/Latin glyph evidence.'},
    latin_evidence_limit_exceeded: {zh: '数字／英文过长，本次保留原图供复核。', en: 'The numeric/Latin line exceeds the evidence limit; review the source crop.'},
    latin_segmentation_uncertain: {zh: '无法可靠分出数字／英文字形，保留原图供复核。', en: 'Numeric/Latin glyphs could not be segmented reliably.'},
    latin_segmentation_incomplete: {zh: '部分数字／英文字形无法可靠分字，字体待确认。', en: 'Some numeric/Latin glyphs could not be segmented reliably.'},
    too_few_distinct_latin_characters: {zh: '不同数字／字母过少，无法可靠区分字体。', en: 'Too few distinct digits or letters to distinguish font families.'},
    low_quality_latin_glyph: {zh: '数字／英文字形过小或图像质量不足。', en: 'Numeric/Latin glyphs are too small or lack sufficient image quality.'},
    unstable_latin_family: {zh: '数字／英文字体候选在不同图像检查中不一致。', en: 'Numeric/Latin family candidates disagree across image checks.'},
    latin_font_outside_reference_gate: {zh: '数字／英文字形与现有参考差异较大，字体待确认。', en: 'Numeric/Latin glyphs differ too much from the available references.'},
    ambiguous_latin_font_families: {zh: '多个数字／英文字体过于接近，保留候选供复核。', en: 'Several numeric/Latin font families are too similar to distinguish reliably.'},
    too_many_regions: {zh: '该图文字区域过多；已保留框，本次超出处理上限。', en: 'This image contains too many text regions. The box is preserved, but this region exceeds the processing limit.'},
    text_unreliable: {zh: '文字未可靠读出，或文字行超出长度限制。', en: 'The text could not be read reliably, or the line exceeds the length limit.'},
    out_of_scope: {zh: '当前模型识别中文字体，数字和英文保留框。', en: 'The current model identifies Chinese fonts; Latin letters and numbers retain detection boxes only.'},
    insufficient_evidence: {zh: '字形证据不足或未通过字体确认门槛。', en: 'Glyph evidence is insufficient or did not pass the font confirmation threshold.'},
    pingfang_supported: {zh: '中文字形通过苹方门槛，逐字候选一致。', en: 'The Chinese glyphs pass the PingFang threshold and agree across character candidates.'},
    consistent_candidate: {zh: '该中文字体在三种图像检查中排名一致，字体名仍为候选。', en: 'This Chinese font ranks consistently across three image checks; the font name remains a candidate.'},
    segmentation_failed: {zh: '部分汉字无法可靠分字，保留原图区域供复核。', en: 'Some Chinese characters could not be segmented reliably; the source region is preserved for review.'}
  };

  function fontReason(value) {
    const reason = String(value.reason || '');
    const coded = reasonCodes[rejectionReason(value)] || reasonCodes[value.reason_code] || reasonCodes[reason];
    if (coded) return coded[language];
    if (!reason) return t('noReason');
    if (language === 'zh') return reason;
    if (reasonTranslations[reason]) return reasonTranslations[reason];
    const missing = reason.match(/^字库尚未收录[：:]\s*(.*)$/);
    if (missing) return `Characters not yet included in the reference library: ${missing[1]}`;
    return /^[\x00-\x7F]+$/.test(reason) ? reason.replaceAll('_', ' ') : t('noReason');
  }

  function show(nextResult) {
    result = nextResult;
    selected = null;
    $('result-section').hidden = false;
    $('original').onload = draw;
    $('original').src = result.image_url || '';
    $('download-png').href = result.annotated_image_url || '';
    $('download-png').setAttribute('aria-disabled', result.annotated_image_url ? 'false' : 'true');
    $('json-output').textContent = JSON.stringify(result, null, 2);
    $('copy-json').disabled = false;
    $('copy-status').textContent = '';
    renderResultText();
    renderList();
    renderDetail();
    draw();
  }

  function renderResultText() {
    if (!result) return;
    const summary = result.summary || {};
    const elapsed = result.timing_seconds?.total ?? result.timing_seconds;
    const identified = (result.regions || []).filter(region => !rejectionReason(font(region)) && ['supported', 'candidate'].includes(font(region).status) && font(region).family).length;
    const parts = [
      {text: t('detected', {count: summary.detected_regions ?? result.regions?.length ?? 0}), tone: 'detected'},
      {text: t('neuralIdentified', {count: identified}), tone: 'supported'}
    ];
    if (isNeural(result)) parts.push({text: t('neuralMethod'), tone: 'neutral'});
    if (Number.isFinite(elapsed)) parts.push({text: t('seconds', {count: Number(elapsed).toFixed(2)}), tone: 'neutral'});
    if (result.model_version) parts.push({text: result.model_version, tone: 'neutral'});
    $('summary').replaceChildren(...parts.map(part => {
      const item = document.createElement('span');
      item.className = `summary-chip ${part.tone}`;
      item.textContent = part.text;
      return item;
    }));
  }

  function draw() {
    const root = $('overlay');
    root.replaceChildren();
    if (!result) return;
    root.setAttribute('viewBox', `0 0 ${result.width} ${result.height}`);
    root.setAttribute('preserveAspectRatio', 'none');
    root.setAttribute('aria-label', t('overlayLabel'));
    for (const region of result.regions || []) {
      const value = font(region);
      const polygon = svg('polygon', {
        class: `quad ${value.status || 'uncertain'}${selected === region.id ? ' selected' : ''}`,
        points: points(region.quad), tabindex: 0, role: 'button', 'aria-label': t('regionLabel', {id: region.id}),
        'data-region-id': region.id, 'aria-pressed': selected === region.id,
        'aria-controls': 'detail-panel', 'aria-expanded': selected === region.id
      });
      polygon.addEventListener('click', () => select(region.id));
      polygon.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(region.id); }
      });
      root.append(polygon);
    }
  }

  function styleMetadata(region, detailed = false) {
    const style = region.text_style;
    if (!style) return null;
    const root = document.createElement('span');
    root.className = detailed ? 'text-style text-style-detail' : 'text-style';
    const rejected = rejectionReason(font(region));
    if (!rejected) {
      const size = document.createElement('span');
      size.className = 'text-size';
      const px = style.font_size_px_estimate;
      const sizeValue = Number.isFinite(px) && px > 0 ? `≈ ${px.toFixed(1).replace(/\.0$/, '')} px` : t('styleUnknown');
      size.textContent = detailed || !Number.isFinite(px) || px <= 0 ? `${t('estimatedSize')}：${sizeValue}` : sizeValue;
      size.setAttribute('aria-label', `${t('estimatedSize')} ${sizeValue}`);
      const range = style.font_size_px_interval;
      if (Array.isArray(range) && range.length === 2 && range.every(Number.isFinite)) {
        size.title = t('sizeRange', {low: range[0].toFixed(1), high: range[1].toFixed(1)});
      }
      root.append(size);
    }
    const color = document.createElement('span');
    color.className = 'text-color';
    const hex = typeof style.text_color_hex === 'string' && /^#[0-9a-f]{6}$/i.test(style.text_color_hex) ? style.text_color_hex.toUpperCase() : null;
    if (detailed) color.append(`${t('textColor')}：`);
    if (hex) {
      const swatch = document.createElement('span');
      swatch.className = 'color-swatch';
      swatch.style.backgroundColor = hex;
      swatch.setAttribute('aria-hidden', 'true');
      color.append(swatch, hex);
    } else color.append(detailed ? t('styleUnknown') : `${t('textColor')}：${t('styleUnknown')}`);
    color.setAttribute('aria-label', `${t('textColor')} ${hex || t('styleUnknown')}`);
    root.append(color);
    if (detailed) {
      const note = document.createElement('small');
      note.className = 'text-style-note';
      note.textContent = t(rejected ? 'colorOnlyNote' : 'styleNote');
      root.append(note);
    }
    return root;
  }

  function renderList() {
    const root = $('regions');
    const previousScroll = root.scrollTop;
    const focusedRegion = root.contains(document.activeElement) ? document.activeElement.dataset.regionId : null;
    root.replaceChildren();
    for (const region of result?.regions || []) {
      const value = font(region);
      const prediction = scoredCandidates(value)[0];
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.regionId = region.id;
      button.setAttribute('aria-pressed', String(selected === region.id));
      button.setAttribute('aria-controls', 'detail-panel');
      button.setAttribute('aria-expanded', String(selected === region.id));
      button.className = `region ${value.status || 'uncertain'}${selected === region.id ? ' selected' : ''}`;
      button.onclick = () => select(region.id, true);
      const tag = document.createElement('span');
      tag.className = `tag ${value.status || 'uncertain'}`;
      tag.textContent = t(statusKey(value.status));
      const text = document.createElement('span');
      const strong = document.createElement('b');
      const small = document.createElement('small');
      strong.textContent = region.id;
      small.className = isNeural(value) ? 'font-prediction' : '';
      small.textContent = rejectionReason(value) ? fontLabel(value) : prediction ? t('closestFont', {family: prediction.family}) : isNeural(value) ? t('noModelScore') : fontLabel(value);
      text.append(strong, small);
      if (prediction) {
        const score = document.createElement('span');
        score.className = 'font-score';
        score.textContent = t('modelScore', {score: prediction.score.toFixed(4)});
        text.append(score);
      }
      const style = styleMetadata(region);
      if (style) text.append(style);
      const thumbnail = document.createElement('span');
      thumbnail.className = 'region-thumbnail';
      if (region.crop_url) {
        const image = new Image();
        image.src = region.crop_url;
        image.alt = t('thumbnailAlt', {id: region.id});
        image.loading = 'lazy';
        image.decoding = 'async';
        image.onerror = () => { thumbnail.textContent = t('cropUnavailable'); };
        thumbnail.append(image);
      } else thumbnail.textContent = t('cropUnavailable');
      text.className = 'region-description';
      const heading = document.createElement('span');
      heading.className = 'region-heading';
      text.replaceChild(heading, strong);
      heading.append(strong, tag);
      button.append(thumbnail, text);
      root.append(button);
    }
    root.scrollTop = previousScroll;
    if (focusedRegion != null) root.querySelector(`[data-region-id="${CSS.escape(String(focusedRegion))}"]`)?.focus({preventScroll: true});
  }

  function renderDetail() {
    const root = $('detail');
    const region = (result?.regions || []).find(item => item.id === selected);
    $('detail-panel').hidden = !region;
    if (!region) {
      root.className = 'empty-state';
      root.textContent = result && !(result.regions || []).length ? t('noRegions') : t('detailEmpty');
      return;
    }
    const value = font(region);
    const candidates = rejectionReason(value) ? [] : isNeural(value) ? scoredCandidates(value) : Array.isArray(value.candidates) ? value.candidates : [];
    const prediction = isNeural(value) ? candidates[0] : null;
    root.className = 'detail-content';
    root.replaceChildren();
    const preview = document.createElement('section');
    preview.className = 'detail-section detail-preview';
    const cropHeading = document.createElement('h3');
    cropHeading.textContent = t('cropTitle');
    preview.append(cropHeading);
    const verdict = document.createElement('section');
    verdict.className = 'detail-section detail-verdict';
    const verdictHeading = document.createElement('h3');
    verdictHeading.textContent = t('fontConclusionTitle');
    verdict.append(verdictHeading);
    if (region.crop_url) {
      const image = new Image();
      image.className = 'detail-image';
      image.src = region.crop_url;
      image.alt = t('cropAlt');
      preview.append(image);
    }
    const title = document.createElement('h2');
    title.textContent = region.id;
    const identity = document.createElement('p');
    const label = document.createElement('strong');
    label.id = 'font-label';
    label.textContent = rejectionReason(value) ? fontLabel(value) : prediction ? t('closestFont', {family: prediction.family}) : isNeural(value) ? t('noModelScore') : value.family || fontLabel(value);
    identity.append(label);
    const badge = document.createElement('span');
    badge.className = `tag ${value.status || 'uncertain'}`;
    badge.textContent = t(statusKey(value.status));
    identity.append(' ', badge);
    const reason = document.createElement('p');
    reason.id = 'font-reason';
    reason.textContent = fontReason(value);
    const note = document.createElement('p');
    note.className = 'muted';
    note.textContent = t(isNeural(value) || value.scope === 'Detected text region' ? 'regionScope' : value.components?.length ? 'scopeMixed' : value.scope === 'Latin letters and digits only' ? 'scopeLatin' : 'scopeNote');
    preview.append(title);
    const style = styleMetadata(region, true);
    if (style) preview.append(style);
    verdict.append(identity);
    if (prediction) {
      const score = document.createElement('p');
      score.id = 'font-score';
      score.className = 'font-score';
      score.textContent = t('modelScore', {score: prediction.score.toFixed(4)});
      verdict.append(score);
    }
    verdict.append(reason, note);
    for (const component of rejectionReason(value) ? [] : value.components || []) {
      const line = document.createElement('p');
      const heading = document.createElement('strong');
      heading.textContent = `${t(component.scope === 'Latin letters and digits only' ? 'latinPart' : 'chinesePart')} · ${fontLabel(component)}`;
      const explanation = document.createElement('span');
      explanation.textContent = fontReason(component);
      line.append(heading, document.createElement('br'), explanation);
      verdict.append(line);
    }
    root.append(preview, verdict);
    if (candidates.length) {
      const distances = document.createElement('div');
      distances.className = 'dist';
      const heading = document.createElement('p');
      heading.textContent = t(isNeural(value) ? 'topScores' : 'topDistances');
      distances.append(heading);
      for (const candidate of candidates.slice(0, 3)) {
        const family = document.createElement('span');
        const distance = document.createElement('span');
        family.textContent = candidate.family;
        const number = isNeural(value) ? candidate.score : candidate.distance;
        distance.textContent = Number.isFinite(number) ? number.toFixed(4) : '—';
        distances.append(family, distance);
      }
      verdict.append(distances);
    }
  }

  function revealInContainer(container, item) {
    if (!container || !item) return;
    const outer = container.getBoundingClientRect();
    const inner = item.getBoundingClientRect();
    const top = outer.top + container.clientTop;
    const bottom = top + container.clientHeight;
    if (inner.top < top) container.scrollTop += inner.top - top;
    else if (inner.bottom > bottom) container.scrollTop += Math.min(inner.top - top, inner.bottom - bottom);
  }

  function select(id, revealInImage = false) {
    if (!(result?.regions || []).some(region => region.id === id)) return;
    const changed = selected !== id;
    selected = id;
    // Keep the existing controls and their focus; only replace the detail pane.
    for (const node of document.querySelectorAll('#overlay [data-region-id], #regions [data-region-id]')) {
      const active = node.dataset.regionId === id;
      node.classList.toggle('selected', active);
      node.setAttribute('aria-pressed', String(active));
      node.setAttribute('aria-expanded', String(active));
    }
    if (changed) {
      renderDetail();
      $('detail').scrollTop = 0;
    }
    revealInContainer($('regions'), $('regions').querySelector('.selected'));
    if (revealInImage) revealInContainer(document.querySelector('.viewer'), $('overlay').querySelector('.quad.selected'));
  }

  function closeDetail() {
    const control = $('regions').querySelector('.selected');
    selected = null;
    for (const node of document.querySelectorAll('#overlay [data-region-id], #regions [data-region-id]')) {
      node.classList.remove('selected');
      node.setAttribute('aria-pressed', 'false');
      node.setAttribute('aria-expanded', 'false');
    }
    renderDetail();
    control?.focus({preventScroll: true});
  }

  function fallbackCopy(text) {
    const previousFocus = document.activeElement;
    const input = document.createElement('textarea');
    input.value = text;
    input.setAttribute('readonly', '');
    input.className = 'clipboard-fallback';
    document.body.append(input);
    input.select();
    input.setSelectionRange(0, input.value.length);
    let copied = false;
    try { copied = document.execCommand('copy'); } finally { input.remove(); previousFocus?.focus({preventScroll: true}); }
    if (!copied) throw Error('copy failed');
  }

  async function copyJson() {
    if (!result) return;
    const raw = JSON.stringify(result, null, 2);
    try {
      if (navigator.clipboard?.writeText) {
        try { await navigator.clipboard.writeText(raw); }
        catch (_) { fallbackCopy(raw); }
      } else fallbackCopy(raw);
      $('copy-status').textContent = t('copied');
    } catch (_) {
      $('copy-status').textContent = t('copyFailed');
      $('json-output').focus();
    }
  }

  function renderFontModel() {
    const download = $('download-model');
    const available = modelInfo?.available === true && modelInfo.download_url === '/api/models/font/download';
    download.setAttribute('aria-disabled', String(!available));
    $('font-model').setAttribute('aria-busy', String(modelLoading));
    $('refresh-model').disabled = modelLoading;
    if (available) download.href = modelInfo.download_url;
    else download.removeAttribute('href');
    $('model-info').textContent = available ? t('modelAvailable', {
      version: modelInfo.version || '—',
      size: Number.isFinite(modelInfo.bytes) ? `${(modelInfo.bytes / 1048576).toFixed(1)} MB` : '—'
    }) : t(modelLoadState);
    const families = Array.isArray(modelInfo?.families) ? modelInfo.families.filter(value => typeof value === 'string') : [];
    $('model-families').hidden = !available || !families.length;
    $('model-families').replaceChildren();
    const sources = modelInfo?.font_sources;
    const grouped = new Set();
    const appendFamilies = (key, names, source = '') => {
      if (!names.length) return;
      const row = document.createElement('p');
      if (source) row.dataset.fontSource = source;
      row.textContent = t(key, {families: names.join(' · ')});
      $('model-families').append(row);
    };
    for (const [source, key] of [['system', 'modelSystemFonts'], ['asset', 'modelAssetFonts']]) {
      const names = families.filter(family => Array.isArray(sources?.[family]) && sources[family].includes(source));
      names.forEach(family => grouped.add(family));
      appendFamilies(key, names, source);
    }
    appendFamilies(grouped.size ? 'modelOtherFonts' : 'modelFamilies', families.filter(family => !grouped.has(family)));
    const pingFangNames = modelInfo?.font_label_groups?.PingFang;
    $('model-label-note').hidden = !available || !families.includes('PingFang') || !Array.isArray(pingFangNames)
      || !pingFangNames.includes('PingFang SC') || !pingFangNames.includes('PingFang TC');
    $('model-label-note').textContent = t('modelPingFangGroup');
    const usage = modelInfo?.usage;
    const hasUsage = available && typeof usage?.install === 'string' && typeof usage?.predict === 'string';
    $('model-usage').hidden = !hasUsage;
    $('model-usage-command').textContent = hasUsage ? `${usage.install}\n${usage.predict}` : '';
  }

  async function loadFontModel() {
    if (modelLoading) return;
    modelLoading = true;
    modelInfo = null;
    modelLoadState = 'modelLoading';
    renderFontModel();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
      modelInfo = await api('/api/models/font', {signal: controller.signal});
      modelLoadState = modelInfo.available === true ? 'modelLoadFailed' : 'modelUnavailable';
    } catch (error) {
      modelInfo = null;
      modelLoadState = error.status === 401 || error.status === 403 ? 'modelLocked' : 'modelLoadFailed';
      if (modelLoadState === 'modelLocked') $('auth-panel').hidden = false;
    } finally {
      clearTimeout(timeout);
      modelLoading = false;
    }
    renderFontModel();
  }

  async function health() {
    try {
      const data = await api('/api/health');
      $('health').dataset.unavailable = 'false';
      $('health').dataset.modelVersion = data.model_version || '';
      $('health').textContent = data.model_version ? t('modelVersion', {version: data.model_version}) : '';
      if (data.authentication_required) $('auth-panel').hidden = false;
    } catch (_) { $('health').dataset.unavailable = 'true'; $('health').textContent = t('healthUnavailable'); }
  }

  async function unlock() {
    const token = $('token').value;
    if (!token) return;
    const error = $('auth-error');
    error.hidden = true;
    $('unlock').disabled = true;
    try {
      await api('/auth', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({token})});
      $('token').value = '';
      $('auth-panel').hidden = true;
      state('unlocked');
      loadFontModel();
    } catch (_) {
      error.textContent = t('authError');
      error.hidden = false;
    } finally { $('unlock').disabled = false; }
  }

  function applyLanguage(nextLanguage, persist = true) {
    language = nextLanguage === 'en' ? 'en' : 'zh';
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    $('language').dataset.language = language;
    $('language').querySelectorAll('[data-language]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.language === language));
    });
    $('language').setAttribute('aria-label', t('language'));
    document.querySelectorAll('[data-i18n]').forEach(node => { node.textContent = t(node.dataset.i18n); });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(node => { node.placeholder = t(node.dataset.i18nPlaceholder); });
    document.querySelectorAll('[data-i18n-aria-label]').forEach(node => { node.setAttribute('aria-label', t(node.dataset.i18nAriaLabel)); });
    $('progress-track').setAttribute('aria-label', t('progressLabel'));
    $('original').alt = t('originalAlt');
    $('overlay').setAttribute('aria-label', t('overlayLabel'));
    $('json-output').setAttribute('aria-label', t('jsonLabel'));
    $('filename').textContent = file ? file.name : t('noFile');
    state(statusState.key, !$('error').hidden, statusState.values);
    const modelVersion = $('health').dataset.modelVersion;
    if (modelVersion) $('health').textContent = t('modelVersion', {version: modelVersion});
    else if ($('health').dataset.unavailable === 'true') $('health').textContent = t('healthUnavailable');
    if (!$('auth-error').hidden) $('auth-error').textContent = t('authError');
    if (progressState && $('error').hidden) updateProgress(progressState.progress, progressState.status, progressState.rawStage, progressState.forceComplete);
    else if (progressState) markProgressError($('error').textContent);
    if (latestSnapshot) updateQueue(latestSnapshot);
    if (reconnectState) state('reconnecting', false, reconnectState);
    if (result) { renderResultText(); draw(); renderList(); renderDetail(); }
    renderFontModel();
    if ($('copy-status').textContent) $('copy-status').textContent = t($('copy-status').textContent === messages.zh.copied || $('copy-status').textContent === messages.en.copied ? 'copied' : 'copyFailed');
    if (persist) { try { localStorage.setItem('flux-glyph-language', language); } catch (_) {} }
  }

  $('unlock').onclick = unlock;
  $('token').addEventListener('keydown', event => { if (event.key === 'Enter') unlock(); });
  $('language').addEventListener('click', event => {
    const button = event.target.closest('[data-language]');
    if (button && button !== $('language') && $('language').contains(button)) applyLanguage(button.dataset.language);
  });
  $('copy-json').addEventListener('click', copyJson);
  $('refresh-model').addEventListener('click', loadFontModel);
  $('close-detail').addEventListener('click', closeDetail);
  $('detail-panel').addEventListener('keydown', event => { if (event.key === 'Escape') closeDetail(); });
  $('file').onchange = event => setFile(event.target.files[0] || null);
  $('drop').addEventListener('click', event => { if (event.target !== $('file')) $('file').click(); });
  $('drop').addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('file').click(); }
  });
  for (const name of ['dragover', 'dragleave', 'drop']) $('drop').addEventListener(name, event => {
    event.preventDefault();
    if (name === 'dragover') $('drop').classList.add('drag');
    else $('drop').classList.remove('drag');
    if (name === 'drop') {
      const dropped = event.dataTransfer.files[0];
      if (dropped && (/^image\//.test(dropped.type) || !dropped.type)) setFile(dropped);
      else state('chooseImage', true);
    }
  });
  $('start').onclick = () => {
    if (!file) return;
    start('/api/jobs', {method: 'POST', headers: {'Content-Type': file.type || 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name)}, body: file});
  };

  applyLanguage(language, false);
  health();
  loadFontModel();
  window.FluxGlyphUI = {show, setFile, start, select, setLanguage: applyLanguage, get: () => ({file, job, result, selected, generation, language, snapshot: latestSnapshot, progress: latestSnapshot?.progress || null, queue: latestSnapshot ? {position: latestSnapshot.queue_position, ahead: latestSnapshot.queue_ahead, total: latestSnapshot.queue_total} : null})};
})();
