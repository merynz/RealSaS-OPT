/* User-facing routing and progressive disclosure only; editing authority remains in Studio. */
(() => {
  const doc=document, timeline=doc.querySelector('#toggleTimeline'),more=doc.querySelector('#advancedTools'),stages=doc.querySelector('#advancedStages');
  const welcome=doc.querySelector('#studioWelcome'),badge=doc.querySelector('#proofBadge');
  function sync(){
    const s=window.RealSaSStudio?.state;
    const valid=Boolean(s?.bundle&&s?.scene?.puppet&&s?.scene?.proof?.passed===true);
    doc.body.classList.toggle('has-puppet',valid);
    if(welcome)welcome.hidden=valid;
    if(s?.bundle)for(const a of doc.querySelectorAll('[data-bundle-nav]')){
      const u=new URL(a.getAttribute('href').split('?')[0],location.href);
      u.searchParams.set('bundle',s.bundle);a.href=u.href;
    }
  }
  badge&&new MutationObserver(sync).observe(badge,{childList:true,subtree:true});
  doc.querySelector('#openBundle')?.addEventListener('click',()=>setTimeout(sync,450));
  doc.querySelector('#focusBundleInput')?.addEventListener('click',()=>{
    const input=doc.querySelector('#bundlePath');
    if(input&&getComputedStyle(input).display!=='none')input.focus();
    else {
      doc.body.classList.add('bundle-input-open');
      input?.focus();
    }
  });
  timeline?.addEventListener('click',()=>{
    const on=!doc.body.classList.contains('timeline-open');
    doc.body.classList.toggle('timeline-open',on);
    timeline.setAttribute('aria-pressed',String(on));
    timeline.textContent=on?'HIDE TIMELINE':'SHOW TIMELINE';
  });
  for(const button of doc.querySelectorAll('.mode'))button.addEventListener('click',()=>{
    if(button.dataset.mode==='animation'&&timeline){
      doc.body.classList.add('timeline-open');timeline.setAttribute('aria-pressed','true');timeline.textContent='HIDE TIMELINE';
    }
    if(more)more.open=false;
  });
  doc.addEventListener('keydown',e=>{if(e.key==='Escape'){if(more)more.open=false;if(stages)stages.open=false;doc.body.classList.remove('bundle-input-open');}});
  doc.addEventListener('pointerdown',e=>{
    if(more?.open&&!more.contains(e.target))more.open=false;
    if(stages?.open&&!stages.contains(e.target))stages.open=false;
  });

  // Transform commands remain real editor operations; expose them on demand.
  const toolbar=doc.querySelector('.viewport-toolbar');
  function organizeToolbar(){
    const group=doc.querySelector('#studioTransformTools');
    if(!toolbar||!group||group.closest('.transform-menu'))return;
    const details=doc.createElement('details');
    details.className='transform-menu';
    details.id='transformMenu';
    const summary=doc.createElement('summary');
    summary.textContent='TRANSFORMS';
    group.replaceWith(details);details.append(summary,group);
  }
  if(toolbar){
    new MutationObserver(organizeToolbar).observe(toolbar,{childList:true});
    organizeToolbar();
  }
  sync();
})();
