/* Mobile-only Studio sidebar controls, no changes to editor state or authority. */
(() => {
  'use strict';
  const buttons = [
    {name:'scene',element:document.querySelector('#toggleScenePanel')},
    {name:'inspector',element:document.querySelector('#toggleInspectorPanel')}
  ];
  if (buttons.some(item=>!item.element)) return;
  const app = document.body;
  const viewport = document.querySelector('.viewport-panel');

  function setPanel(name) {
    const next = app.dataset.studioPanel === name ? '' : name;
    app.dataset.studioPanel = next;
    buttons.forEach(item => item.element.setAttribute('aria-expanded', String(item.name === next)));
  }
  buttons.forEach(item => item.element.addEventListener('click',()=>setPanel(item.name)));
  viewport?.addEventListener('pointerdown',()=>{
    if(app.dataset.studioPanel) setPanel(app.dataset.studioPanel);
  });
  document.addEventListener('keydown',(event)=>{
    if(event.key === 'Escape' && app.dataset.studioPanel) setPanel(app.dataset.studioPanel);
  });
})();
