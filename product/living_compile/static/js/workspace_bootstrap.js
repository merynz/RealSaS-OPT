const params = new URLSearchParams(window.location.search);
const bundle = String(params.get('bundle') || '').trim();

if (bundle) {
  const input = document.querySelector('#bundlePath');
  const open = document.querySelector('#openBundle');
  if (input && open) {
    input.value = bundle;
    window.addEventListener('load', () => {
      open.click();
    }, {once: true});
  }
}
