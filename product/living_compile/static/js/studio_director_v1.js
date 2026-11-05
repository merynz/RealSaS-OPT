/* BUSTED / Studio Director 1
 * Routes to existing qualified Studio tools. PUPPET and MOTION are workspaces
 * on the SAME editable product, never fake canvases or unqualified preview.
 */
(() => {
  'use strict';
  const dock=document.querySelector('#studioDirector');
  if(!dock)return;
  const $=s=>document.querySelector(s);
  const workspaceButtons=[...dock.querySelectorAll('[data-director-workspace]')];
  const commandButtons=[...dock.querySelectorAll('[data-director-command]')];
  const toolCommands=new Set(['select','translate','rotate','scale']);
  const legacyModes={rig:'rig',mesh:'mesh',skin:'weights',attach:'bindings',ik:'ik'};
  const params=new URLSearchParams(location.search);
  let studio=null;
  let workspace=params.get('workspace')==='motion'?'motion':'puppet';
  let lastState='';
  function ready(){
    const s=studio?.state;
    return Boolean(s?.bundle&&s?.scene?.puppet&&s?.scene?.proof?.passed===true);
  }
  function clip(){
    const s=studio?.state;
    return (s?.ed?.animation||[]).find(item=>item.clip_id===s.clip)||null;
  }
  function availableControls(){
    const s=studio?.state;
    return (s?.scene?.puppet?.rig?.animator_controls||[]).filter(x=>x.view_id==='V'+s?.view).length;
  }
  function click(selector){
    const el=$(selector);
    if(!el||el.disabled)return false;
    el.click();return true;
  }
  function chooseMode(name){
    const button=[...document.querySelectorAll('button.mode[data-mode]')].find(x=>x.dataset.mode===name);
    if(button)button.click();
    return Boolean(button);
  }
  function showTimeline(value){
    if(document.body.classList.contains('timeline-open')!==value)click('#toggleTimeline');
  }
  function switchWorkspace(name,{updateURL=true}={}){
    if(name!=='puppet'&&name!=='motion')return;
    workspace=name;
    document.body.dataset.studioWorkspace=workspace;
    chooseMode(workspace==='motion'?'animation':'rig');
    showTimeline(workspace==='motion');
    studio?.toolContext?.setTool('select');
    if(updateURL){
      const url=new URL(location.href);
      if(workspace==='motion')url.searchParams.set('workspace','motion');
      else url.searchParams.delete('workspace');
      history.replaceState(null,'',url);
    }
    render(true);
  }
  function run(command){
    if(command==='open'){click('#focusBundleInput');return}
    if(!ready())return;
    if(Object.prototype.hasOwnProperty.call(legacyModes,command)){
      switchWorkspace('puppet');
      chooseMode(legacyModes[command]);
    }else if(toolCommands.has(command)){
      studio?.toolContext?.setTool(command);
    }else if(command==='new-clip'){
      switchWorkspace('motion');
      click('#newClip');
    }else if(command==='add-key'){
      if(workspace==='motion'&&clip())click('#addKey');
    }else if(command==='play'){
      if(workspace==='motion'&&clip())click('#play');
    }else if(command==='stop'){
      click('#stop');
    }else if(command==='fit'){
      click('#fitView');
    }else if(command==='save'){
      click('#saveLayer');
    }
    render(true);
  }
  workspaceButtons.forEach(b=>b.addEventListener('click',()=>switchWorkspace(b.dataset.directorWorkspace)));
  commandButtons.forEach(b=>b.addEventListener('click',()=>run(b.dataset.directorCommand)));
  function render(force=false){
    if(!studio)return;
    const s=studio.state;
    const loaded=ready(),currentClip=clip(),controlCount=availableControls();
    const selected=studio.selection?.primary();
    const selectedName=selected?.control_id||selected?.mesh_id||s.sel?.control||'NONE';
    const mode=s?.mode||'rig',tool=studio.toolContext?.tool||'select';
    const state=JSON.stringify([loaded,workspace,currentClip?.clip_id,controlCount,selectedName,mode,tool,s?.view,studio.dirty?.status,document.body.classList.contains('timeline-open')]);
    if(!force&&state===lastState)return;
    lastState=state;
    dock.dataset.ready=String(loaded);
    dock.dataset.workspace=workspace;
    const title=$('#directorStatus'),selection=$('#directorSelection'),instruction=$('#directorInstruction');
    if(title)title.textContent=loaded?'QUALIFIED PUPPET':'NO PUPPET LOADED';
    if(selection)selection.textContent=loaded?('VIEW V'+s.view+' / '+String(selectedName).slice(0,36)):'WAITING FOR QUALIFIED BUNDLE';
    if(instruction){
      if(!loaded)instruction.textContent='AWAKEN a character, then open its proof-qualified Studio bundle.';
      else if(workspace==='puppet')instruction.textContent='SETUP: rig, mesh, skin and attachments. Changes require requalification.';
      else if(!controlCount)instruction.textContent='This view has no animator controls. Direct posing is unavailable.';
      else if(!currentClip)instruction.textContent='Create a USER CLIP, select a control, then move / rotate / scale to key.';
      else instruction.textContent='POSE: transform controls at the playhead. Changes write real user keyframes.';
    }
    workspaceButtons.forEach(b=>{
      const active=b.dataset.directorWorkspace===workspace;
      b.classList.toggle('active',active);
      b.setAttribute('aria-pressed',String(active));
    });
    for(const button of commandButtons){
      const command=button.dataset.directorCommand;
      const needsClip=['add-key','play'].includes(command);
      const needsControl=needsClip;
      const isMotion=['new-clip','add-key','play','stop'].includes(command);
      const isPuppet=Object.prototype.hasOwnProperty.call(legacyModes,command);
      button.hidden=(isMotion&&workspace!=='motion')||(isPuppet&&workspace!=='puppet');
      button.disabled=command!=='open'&&(!loaded||(needsClip&&(!currentClip||(needsControl&&!controlCount))));
      button.classList.toggle('active',toolCommands.has(command)&&tool===command);
      button.setAttribute('aria-pressed',String(toolCommands.has(command)&&tool===command));
    }
    dock.classList.toggle('director-has-clip',Boolean(currentClip));
  }
  function connect(){
    studio=window.RealSaSStudio||null;
    if(!studio||!studio.toolContext){requestAnimationFrame(connect);return}
    studio.toolContext.addEventListener('change',()=>render(true));
    studio.selection?.addEventListener('change',()=>render(true));
    studio.dirty?.addEventListener('change',()=>render(true));
    switchWorkspace(workspace,{updateURL:false});
    let last=0;
    function watch(now){
      if(now-last>320){last=now;if(!document.hidden)render()}
      requestAnimationFrame(watch);
    }
    requestAnimationFrame(watch);
  }
  connect();
})();
