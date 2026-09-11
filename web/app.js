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
      tagline: '上传截图，查看每个文字区域的字体结果与原图证据。', language: '语言', apiExample: 'API 示例',
      unlockTitle: '解锁访问', unlockHelp: '此服务需要访问令牌。令牌只在本次页面会话中用于设置安全 Cookie，不会保存在浏览器存储中。', tokenPlaceholder: '输入访问令牌', unlock: '解锁',
      uploadTitle: '上传图片', uploadHelp: '支持 PNG、JPG、WebP。字体结果仅针对框内中文；英文和数字保留检测框。', dropPrompt: '选择图片或拖入这里', noFile: '尚未选择图片', start: '开始识别', downloadPng: '下载标注 PNG',
      waiting: '等待上传', selected: '已选择 {name}，等待开始识别', creating: '正在创建任务…', queued: '排队中…', running: '正在识别…', complete: '识别完成', failed: '识别失败', pollFailed: '轮询失败：{message}', reconnecting: '连接暂时中断，{seconds} 秒后继续查询当前任务…', jobGone: '任务已不存在，请再次开始识别。', authRequired: '登录已失效，请重新解锁。', chooseImage: '请选择单张图片',
      preparing: '正在准备图片…', detecting: '正在框选文字区域…', recognizing: '正在识读文字…', matching: '正在匹配字体…', annotating: '正在生成字体标注图片…', finalizing: '正在整理结果…', progressLabel: '阶段进度', progressCount: '{current}/{total}', queuePosition: '排队第 {position} 位，前方 {ahead} 个任务', queueNext: '即将开始处理', busy: '当前排队任务较多，请稍后再试。图片仍保留，可直接再次开始识别。', invalidImage: '图片无法读取。请上传 8 MB 内、1200 万像素内的 PNG、JPG 或 WebP 静态图。', imageTooLarge: '图片超过 8 MB 上传限制，请选择较小的图片。', uploadFailed: '任务创建失败，请检查连接后再次开始识别。', restarted: '服务已重启，请再次开始识别。',
      resultsTitle: '识别结果', detailEmpty: '点击图片中的文字框或区域列表查看字体证据。', noRegions: '未检测到文字区域。请使用清晰、未裁切的截图重试。', originalAlt: '后端已校正方向的原图', overlayLabel: '检测到的文字区域', regionLabel: '文字区域 {id}', cropAlt: '所选文字区域裁图', glyphAlt: '单字 {character}',
      copyJson: '复制 JSON', copied: 'JSON 已复制', copyFailed: '无法自动复制，请在 JSON 区域中手动选择复制。', jsonLabel: '格式化 JSON',
      detected: '检测 {count}', chinese: '中文 {count}', pingfang: '苹方支持 {count}', seconds: '{count} 秒',
      supported: '支持', candidate: '候选', uncertain: '待确认', outOfScope: '不支持', unknownText: '未识别文字', unknownFont: '待确认', noReason: '没有进一步说明', scopeNote: '字体结果仅针对框内中文；英文和数字尚未判定。', topDistances: 'Top 3 原始距离（距离不是概率）', glyphEvidence: '原图单字证据',
      modelVersion: '模型版本：{version}', healthUnavailable: '服务状态暂不可用', unlocked: '已解锁，可以上传图片', authError: '令牌无效或服务拒绝访问，请重试。',
      apiHelp: '上传支付宝图片，完成后直接返回字体识别 JSON。', apiAsync: '使用 ?wait=false 先返回任务 ID，再轮询 GET /api/jobs/{id}。仍兼容 POST /api/predict?wait=true。', apiDocs: '完整 API 说明'
    },
    en: {
      tagline: 'Upload a screenshot to inspect font results and source evidence for each text region.', language: 'Language', apiExample: 'API examples',
      unlockTitle: 'Unlock access', unlockHelp: 'This service requires an access token. It is used only to set a secure cookie for this page session and is never saved in browser storage.', tokenPlaceholder: 'Enter access token', unlock: 'Unlock',
      uploadTitle: 'Upload image', uploadHelp: 'Supports PNG, JPG, and WebP. Font results cover Chinese text only; Latin letters and numbers keep their detection boxes.', dropPrompt: 'Choose an image or drop it here', noFile: 'No image selected', start: 'Start recognition', downloadPng: 'Download annotated PNG',
      waiting: 'Waiting for an image', selected: '{name} selected. Ready to start.', creating: 'Creating job…', queued: 'Waiting in queue…', running: 'Recognizing…', complete: 'Recognition complete', failed: 'Recognition failed', pollFailed: 'Status check failed: {message}', reconnecting: 'Connection interrupted. Checking this job again in {seconds} seconds…', jobGone: 'This job is no longer available. Start recognition again.', authRequired: 'Your access session expired. Unlock the service again.', chooseImage: 'Please choose one image',
      preparing: 'Preparing image…', detecting: 'Detecting text regions…', recognizing: 'Reading text…', matching: 'Matching fonts…', annotating: 'Generating annotated image…', finalizing: 'Finalizing results…', progressLabel: 'Stage progress', progressCount: '{current}/{total}', queuePosition: 'Queue position {position} · {ahead} job(s) ahead', queueNext: 'Starting soon', busy: 'The queue is currently full. Try again shortly; your selected image is still ready.', invalidImage: 'The image could not be read. Upload a static PNG, JPG, or WebP under 8 MB and 12 megapixels.', imageTooLarge: 'The image exceeds the 8 MB upload limit. Choose a smaller image.', uploadFailed: 'The job could not be created. Check your connection and start recognition again.', restarted: 'The service restarted before this job finished. Start recognition again.',
      resultsTitle: 'Recognition results', detailEmpty: 'Select a box in the image or a region in the list to inspect its font evidence.', noRegions: 'No text regions were found. Try a clear, uncropped screenshot.', originalAlt: 'Source image with orientation corrected by the server', overlayLabel: 'Detected text regions', regionLabel: 'Text region {id}', cropAlt: 'Crop of the selected text region', glyphAlt: 'Glyph {character}',
      copyJson: 'Copy JSON', copied: 'JSON copied', copyFailed: 'Automatic copy failed. Select the formatted JSON and copy it manually.', jsonLabel: 'Formatted JSON',
      detected: 'Detected {count}', chinese: 'Chinese {count}', pingfang: 'PingFang supported {count}', seconds: '{count} sec',
      supported: 'Supported', candidate: 'Candidate', uncertain: 'Review', outOfScope: 'Unsupported', unknownText: 'Unrecognized text', unknownFont: 'Review needed', noReason: 'No further explanation is available.', scopeNote: 'Font results cover Chinese text only; Latin letters and numbers are not classified.', topDistances: 'Top 3 raw distances (distance is not probability)', glyphEvidence: 'Source glyph evidence',
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
    $('download-png').removeAttribute('href');
    $('download-png').setAttribute('aria-disabled', 'true');
  }

  function resetProgress() {
    progressState = null;
    latestSnapshot = null;
    $('progress-panel').hidden = true;
    $('progress-panel').classList.remove('progress-error');
    $('progress-panel').classList.remove('progress-complete');
    const track = $('progress-track');
    track.classList.remove('indeterminate');
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
    if (/匹配字体/.test(human)) return 'matching';
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
    panel.classList.remove('progress-complete');
    track.classList.remove('indeterminate');
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
  const statusKey = value => ({supported: 'supported', candidate: 'candidate', uncertain: 'uncertain', out_of_scope: 'outOfScope'})[value] || 'uncertain';

  function fontLabel(value) {
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
    const coded = reasonCodes[value.reason_code] || reasonCodes[reason];
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
    $('original').src = result.image_url || '';
    $('original').onload = draw;
    $('download-png').href = result.annotated_image_url || '';
    $('download-png').setAttribute('aria-disabled', result.annotated_image_url ? 'false' : 'true');
    $('json-output').textContent = JSON.stringify(result, null, 2);
    $('copy-json').disabled = false;
    $('copy-status').textContent = '';
    renderResultText();
    renderList();
    renderDetail();
  }

  function renderResultText() {
    if (!result) return;
    const summary = result.summary || {};
    const elapsed = result.timing_seconds?.total ?? result.timing_seconds;
    const parts = [
      t('detected', {count: summary.detected_regions ?? result.regions?.length ?? 0}),
      t('chinese', {count: summary.chinese_regions ?? '—'}),
      t('pingfang', {count: summary.pingfang_supported ?? '—'})
    ];
    if (Number.isFinite(elapsed)) parts.push(t('seconds', {count: Number(elapsed).toFixed(2)}));
    if (result.model_version) parts.push(result.model_version);
    $('summary').textContent = parts.join(' · ');
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
        points: points(region.quad), tabindex: 0, role: 'button', 'aria-label': t('regionLabel', {id: region.id})
      });
      polygon.addEventListener('click', () => select(region.id));
      polygon.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(region.id); }
      });
      root.append(polygon);
    }
  }

  function renderList() {
    const root = $('regions');
    root.replaceChildren();
    for (const region of result?.regions || []) {
      const value = font(region);
      const button = document.createElement('button');
      button.className = `region${selected === region.id ? ' selected' : ''}`;
      button.onclick = () => select(region.id);
      const tag = document.createElement('span');
      tag.className = `tag ${value.status || 'uncertain'}`;
      tag.textContent = t(statusKey(value.status));
      const text = document.createElement('span');
      const strong = document.createElement('b');
      const small = document.createElement('small');
      strong.textContent = region.text || t('unknownText');
      small.textContent = fontLabel(value);
      text.append(strong, small);
      button.append(tag, text);
      root.append(button);
    }
  }

  function glyphReason(glyph) {
    const code = String(glyph.reason || glyph.status || '').toLowerCase();
    const translations = language === 'zh' ? {
      ok: '已提取原图字形', reference_character_absent: '字库暂未收录此字', no_low_ink_boundary: '字形相连，暂未可靠分开', low_confidence: '文字识读不稳定', low_ctc_token_confidence: '文字位置不确定', foreground_touches_roi_edge: '字形触及裁图边缘', edge: '字形触及裁图边缘', insufficient_ink: '笔画信息不足'
    } : {
      ok: 'Source glyph extracted', reference_character_absent: 'Character not yet in the reference library', no_low_ink_boundary: 'Connected glyphs could not be separated reliably', low_confidence: 'Text recognition is unstable', low_ctc_token_confidence: 'Character position is uncertain', foreground_touches_roi_edge: 'Glyph touches the crop edge', edge: 'Glyph touches the crop edge', insufficient_ink: 'Insufficient stroke information'
    };
    if (translations[code]) return translations[code];
    if (/ctc_order_refined|connected_foreground_gaps/.test(code)) return translations.ok;
    if (/(?:missing|invalid|outside).*ctc|ctc.*(?:missing|invalid|outside)/.test(code)) return language === 'zh' ? '文字位置不确定' : 'Character position is uncertain';
    if (/edge|border/.test(code)) return translations.edge;
    if (/insufficient.*ink|low.*ink/.test(code)) return translations.insufficient_ink;
    return ({supported: translations.ok, candidate: language === 'zh' ? '字体候选待确认' : 'Font candidate needs review', uncertain: language === 'zh' ? '文字识读不稳定' : 'Text recognition is unstable', out_of_scope: language === 'zh' ? '字体库暂不支持' : 'Not supported by the font library'})[glyph.status] || t('noReason');
  }

  function renderDetail() {
    const root = $('detail');
    const region = (result?.regions || []).find(item => item.id === selected);
    if (!region) {
      root.className = 'empty-state';
      root.textContent = result && !(result.regions || []).length ? t('noRegions') : t('detailEmpty');
      return;
    }
    const value = font(region);
    root.className = '';
    root.replaceChildren();
    if (region.crop_url) {
      const image = new Image();
      image.className = 'detail-image';
      image.src = region.crop_url;
      image.alt = t('cropAlt');
      root.append(image);
    }
    const title = document.createElement('h2');
    title.textContent = region.text || t('unknownText');
    const identity = document.createElement('p');
    const label = document.createElement('strong');
    label.id = 'font-label';
    label.textContent = fontLabel(value);
    identity.append(label, ` · ${t(statusKey(value.status))}`);
    const reason = document.createElement('p');
    reason.id = 'font-reason';
    reason.textContent = fontReason(value);
    const note = document.createElement('p');
    note.className = 'muted';
    note.textContent = t('scopeNote');
    root.append(title, identity, reason, note);
    if (value.candidates?.length) {
      const distances = document.createElement('div');
      distances.className = 'dist';
      const heading = document.createElement('p');
      heading.textContent = t('topDistances');
      distances.append(heading);
      for (const candidate of value.candidates.slice(0, 3)) {
        const family = document.createElement('span');
        const distance = document.createElement('span');
        family.textContent = candidate.family;
        distance.textContent = Number(candidate.distance).toFixed(4);
        distances.append(family, distance);
      }
      root.append(distances);
    }
    if (region.glyphs?.length) {
      const heading = document.createElement('p');
      heading.textContent = t('glyphEvidence');
      root.append(heading);
      for (const glyph of region.glyphs) {
        const row = document.createElement('div');
        row.className = 'glyph';
        if (glyph.crop_url) {
          const image = new Image();
          image.src = glyph.crop_url;
          image.alt = t('glyphAlt', {character: glyph.character});
          row.append(image);
        }
        const text = document.createElement('span');
        const strong = document.createElement('b');
        const small = document.createElement('small');
        strong.textContent = `${glyph.character} · ${glyph.family_candidate || t('unknownFont')}`;
        small.textContent = glyphReason(glyph);
        text.append(strong, small);
        row.append(text);
        root.append(row);
      }
    }
  }

  function select(id) {
    selected = id;
    draw();
    renderList();
    renderDetail();
  }

  function fallbackCopy(text) {
    const input = document.createElement('textarea');
    input.value = text;
    input.setAttribute('readonly', '');
    input.className = 'clipboard-fallback';
    document.body.append(input);
    input.select();
    input.setSelectionRange(0, input.value.length);
    let copied = false;
    try { copied = document.execCommand('copy'); } finally { input.remove(); }
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
  window.FluxGlyphUI = {show, setFile, start, select, setLanguage: applyLanguage, get: () => ({file, job, result, selected, generation, language, snapshot: latestSnapshot, progress: latestSnapshot?.progress || null, queue: latestSnapshot ? {position: latestSnapshot.queue_position, ahead: latestSnapshot.queue_ahead, total: latestSnapshot.queue_total} : null})};
})();
