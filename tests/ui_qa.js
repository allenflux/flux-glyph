const fs = require('fs');
const assert = require('assert');

const html = fs.readFileSync(__dirname + '/../web/index.html', 'utf8');
const js = fs.readFileSync(__dirname + '/../web/app.js', 'utf8');
const css = fs.readFileSync(__dirname + '/../web/style.css', 'utf8');

assert(!/(?:src|href)=["']https?:\/\//.test(html), 'frontend must not require an external CDN');
for (const contract of ['/api/jobs', 'X-Filename', 'createElementNS', 'annotated_image_url', 'image_url', 'crop_url', 'timing_seconds', 'value.reason', '/api/predict?wait=true', '/api/health', 'authentication_required', 'model_version', '/auth']) {
  assert((html + js).includes(contract), `missing API/UI contract: ${contract}`);
}

for (const id of ['file', 'start', 'overlay', 'original', 'detail', 'download-png', 'language', 'progress-panel', 'progress-track', 'progress-bar', 'progress-label', 'progress-count', 'queue-status', 'json-output', 'copy-json', 'copy-status']) {
  assert(new RegExp(`id=["']${id}["']`).test(html), `missing stable UI id: #${id}`);
}
assert(!/id=["']download-json["']/.test(html), 'JSON download control must be removed');
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
assert(/max-height:72vh/.test(css) && /font-family:ui-monospace/.test(css), 'JSON panel must be bounded and monospace');
for (const token of ['@media(max-width:720px)', 'border-radius:4px', 'background:var(--soft)']) {
  assert(css.includes(token), `reference-style token absent: ${token}`);
}

console.log(JSON.stringify({result: 'passed', checks: ['same-origin-only', 'job-upload-contract', 'bilingual-ui', 'real-progress-and-queue', 'inline-json-copy', 'native-svg-contract', 'png-download', 'font-evidence-fields', 'responsive-reference-style']}));
