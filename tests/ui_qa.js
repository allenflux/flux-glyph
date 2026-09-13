const fs = require('fs');
const assert = require('assert');

const html = fs.readFileSync(__dirname + '/../web/index.html', 'utf8');
const js = fs.readFileSync(__dirname + '/../web/app.js', 'utf8');
const css = fs.readFileSync(__dirname + '/../web/style.css', 'utf8');

assert(!/(?:src|href)=["']https?:\/\//.test(html), 'frontend must not require an external CDN');
for (const contract of ['/api/jobs', 'X-Filename', 'createElementNS', 'annotated_image_url', 'image_url', 'crop_url', 'timing_seconds', 'value.reason', '/api/predict?wait=true', '/api/health', 'authentication_required', 'model_version', '/auth', '/api/models/font', '/api/models/font/download', 'region_neural_network']) {
  assert((html + js).includes(contract), `missing API/UI contract: ${contract}`);
}

for (const id of ['file', 'start', 'overlay', 'original', 'detail', 'download-png', 'language', 'progress-panel', 'progress-track', 'progress-bar', 'progress-label', 'progress-count', 'queue-status', 'json-output', 'copy-json', 'copy-status']) {
  assert(new RegExp(`id=["']${id}["']`).test(html), `missing stable UI id: #${id}`);
}
assert(!/id=["']download-json["']/.test(html), 'JSON download control must be removed');
for (const id of ['ocr-output', 'ocr-title', 'copy-ocr', 'ocr-count']) {
  assert(!new RegExp(`id=["']${id}["']`).test(html), `retired OCR control remains: #${id}`);
}
assert(!/function (?:renderOcr|copyOcr|ocrConfidence|glyphReason)\(/.test(js), 'region UI must not restore OCR or single-glyph evidence');
assert(js.includes('strong.textContent = region.id') && js.includes('title.textContent = region.id'), 'region titles must use IDs without recognized text');
assert(js.includes("thumbnail.className = 'region-thumbnail'") && css.includes('.region-thumbnail img'), 'region crops must be visible in the list');
assert(js.includes('candidate.score') && js.includes('isNeural(value)'), 'region neural candidates must show model scores');
for (const label of ['最接近：', 'Closest match:', '模型评分', 'Model score', '未生成评分', 'No model score']) {
  assert(js.includes(label), `missing bilingual neural prediction presentation: ${label}`);
}
for (const hook of ['font-prediction', 'font-score', 'font-label', 'font-reason']) {
  assert(js.includes(hook), `missing stable neural prediction display hook: ${hook}`);
}
assert(css.includes('.font-prediction') && css.includes('.font-score'), 'prediction and score must have responsive presentation styles');
for (const label of ['未知字体', 'Unknown font', 'unknown_font_rejected', 'invalid_rejection_output']) {
  assert(js.includes(label), `missing explicit unknown-font presentation: ${label}`);
}
for (const label of ['字体存在分歧', 'Font predictions disagree', 'neural_model_disagreement', 'verifier_font_out_of_scope',
  'invalid_verifier_output', '主网络', 'Primary network', '复核网络', 'Verifier network']) {
  assert(js.includes(label), `missing explicit consensus presentation: ${label}`);
}
for (const id of ['font-model', 'download-model', 'refresh-model', 'model-info', 'model-families', 'model-label-note', 'model-usage-command']) {
  assert(new RegExp(`id=["']${id}["']`).test(html), `missing standalone model control: #${id}`);
}
assert(js.includes("modelInfo.download_url === '/api/models/font/download'"), 'model downloads must use the fixed same-origin authenticated route');
assert(!html.includes('data-font-mode') && !js.includes('mode=') && !js.includes('setFontMode'), 'a single model must not offer a platform selector or route by mode');
assert(html.includes('id="model-preview"') && html.includes('href="/docs#font-model"'), 'experimental model notice must link to its usage and evaluation');
assert(js.includes("modelInfo?.font_mode === 'unified'") && js.includes('families.length') && js.includes('modelScope'), 'joint coverage and public class count must derive from actual metadata');
assert(js.includes("modelInfo?.release_tier === 'experimental'") && js.includes('尚未通过稳定版验收') && js.includes('Stable validation has not passed'), 'preview evidence must control bilingual notices without claiming stable success');
assert(js.includes('modelInfo?.usage') && js.includes("$('model-usage-command').textContent"), 'model usage must be rendered safely from API data');
assert(js.includes('modelInfo?.font_sources') && js.includes('modelInfo?.font_label_groups?.PingFang'), 'font source categories and label grouping must come from model metadata');
for (const label of ['系统内置字体', '应用自带字体', 'Built-in system fonts', 'App-bundled fonts', '未细分地区版本', 'regional variants are not classified separately']) {
  assert(js.includes(label), `missing bilingual font provenance description: ${label}`);
}
assert(html.indexOf('id="font-model"') < html.indexOf('class="upload-help"') && html.indexOf('id="font-model"') < html.indexOf('id="result-section"'), 'font download must precede upload and results');
assert(/<details id="model-usage"/.test(html) && !/<details id="model-usage"[^>]* open/.test(html), 'local usage must initially be compact and expandable');
assert(js.includes("$('refresh-model').addEventListener('click', loadFontModel)") && js.includes('controller.abort()'), 'model availability needs a bounded request and explicit retry');
assert(/id="language"[^>]*role="group"/.test(html), 'language switch must be an accessible group');
assert(/button[^>]*data-language="zh"[^>]*aria-pressed="true"[^>]*>中文<\/button>/.test(html), 'language switch must provide Chinese');
assert(/button[^>]*data-language="en"[^>]*aria-pressed="false"[^>]*>EN<\/button>/.test(html), 'language switch must provide English');
assert(/JSON\.stringify\(result, null, 2\)/.test(js), 'result JSON must be formatted in the browser');
assert(/\$\('json-output'\)\.textContent\s*=/.test(js), 'formatted JSON must use textContent');
assert(/navigator\.clipboard\?\.writeText/.test(js) && /document\.execCommand\('copy'\)/.test(js), 'copy must include Clipboard API and public-HTTP fallback');
assert(/aria-live="polite"/.test(html) && /id="copy-status"/.test(html), 'copy feedback must be announced');

assert(/typeof data\.percent === 'number'/.test(js), 'progress may only become determinate for a numeric backend percent');
assert(/removeAttribute\('aria-valuenow'\)/.test(js), 'indeterminate progress must omit aria-valuenow');
assert(/forceComplete \? 100/.test(js), 'completed jobs must display 100 percent');
assert(/queue_position/.test(js) && /queue_ahead/.test(js), 'real queue position must be rendered');
assert(/snapshot: latestSnapshot/.test(js), 'latest backend snapshot must remain inspectable for browser QA');
assert(/Math\.min\(10000, 2000/.test(js) && /setTimeout\(\(\) => poll\(activeGeneration\), delay\)/.test(js), 'GET polling must retry with capped backoff without submitting a duplicate job');
assert(/status !== 'queued'/.test(js), 'queued zero percent must remain indeterminate');
assert(/error\.status === 413 \? 'imageTooLarge'/.test(js) && /error\.status === 400 \|\| error\.status === 415/.test(js), 'upload size and format errors must have localized messages');
assert(css.includes('.progress-track.indeterminate') && css.includes('@media(prefers-reduced-motion:reduce)'), 'progress animation and reduced-motion behavior are required');
assert(/transition:none!important;animation:none!important/.test(css), 'reduced-motion mode must stop transitions and animations');

const storageWrites = [...js.matchAll(/localStorage\.setItem\(\s*['"]([^'"]+)['"]/g)].map(match => match[1]);
assert.deepStrictEqual(storageWrites, ['flux-glyph-language'], 'only the language preference may be persisted');
assert(!/(?:localStorage|sessionStorage)\.setItem\([^\n]*(?:token|auth)/i.test(js), 'access tokens must never be persisted');
assert(/\.data-column\{[^}]*grid-template-columns:minmax\(0,1fr\)[^}]*grid-template-rows:/.test(css) && /#json-output\{[^}]*flex:1[^}]*overflow:auto/.test(css) && /font-family:ui-monospace/.test(css), 'JSON must use a full-width row with scrollable monospace content');
for (const id of ['detail-panel', 'close-detail']) assert(new RegExp(`id=["']${id}["']`).test(html), `missing collapsible detail control: #${id}`);
assert(/id="detail-panel"[^>]*hidden/.test(html), 'font detail must be collapsed before a region is selected');
for (const token of ['@media(max-width:760px)', 'border-radius:4px', 'background:var(--soft)']) {
  assert(css.includes(token), `reference-style token absent: ${token}`);
}

console.log(JSON.stringify({result: 'passed', checks: ['same-origin-only', 'job-upload-contract', 'bilingual-ui', 'real-progress-and-queue', 'inline-json-copy', 'native-svg-contract', 'png-download', 'region-crops-without-ocr', 'region-neural-scores', 'authenticated-model-download', 'responsive-reference-style']}));
