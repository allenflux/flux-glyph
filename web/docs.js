(() => {
  'use strict';
  const picker = document.getElementById('language');
  function apply(language) {
    document.documentElement.lang = language === 'en' ? 'en' : 'zh-CN';
    picker.dataset.language = language;
    picker.setAttribute('aria-label', language === 'en' ? 'Language' : '语言');
    picker.querySelectorAll('button[data-language]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.language === language));
    });
    document.querySelectorAll('[data-lang]').forEach(node => { node.hidden = node.dataset.lang !== language; });
  }
  let language = 'zh';
  try { if (localStorage.getItem('flux-glyph-language') === 'en') language = 'en'; } catch (_) {}
  apply(language);
  picker.addEventListener('click', event => {
    const button = event.target.closest('button[data-language]');
    if (!button || !picker.contains(button)) return;
    apply(button.dataset.language);
    try { localStorage.setItem('flux-glyph-language', button.dataset.language); } catch (_) {}
  });
})();
