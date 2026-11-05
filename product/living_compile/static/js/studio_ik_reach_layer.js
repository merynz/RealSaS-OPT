/* BUSTED / Studio IK Reach v1
 * Real two-segment inverse-kinematic math, authored onto user animation keys.
 * Deliberately NOT a Mechanics-approved runtime solver: Studio preview only.
 * A qualified puppet, explicit user clip, valid authored IK constraint and
 * one-to-one animator→deform-joint mapping are strict preconditions.
 */
import {solveTwoBoneReach,reachableTwoBoneConstraints,worldJointAnchor} from './studio_ik_chain_core.js';
import {normalizePose} from './studio_control_semantics.js';
const $=selector=>document.querySelector(selector);
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const dist=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
function boot(){
  const studio=window.RealSaSStudio;
  const viewport=$('#viewport'),panel=$('.viewport-panel');
  if(!studio||!studio.toolContext||!studio.timeline||!studio.refreshAnimationPreview||!viewport||!panel){
    requestAnimationFrame(boot);
    return;
  }
  const S=studio.state,tools=studio.toolContext,timeline=studio.timeline;
  const layer=document.createElement('canvas');
  layer.id='studioIKReachOverlay';
  layer.className='studio-ik-reach-overlay';
  layer.setAttribute('aria-label','Two bone IK reach authoring overlay');
  layer.setAttribute('role','presentation');
  panel.append(layer);
  const ctx=layer.getContext('2d');
  let drag=null,saveTimer=0,lastDraw=0,previousMessage='',cache=[];
  let lastCanvasWidth=0,lastCanvasHeight=0,lastCanvasLeft=NaN,lastCanvasTop=NaN;
  const note=document.createElement('div');
  note.id='studioIKReachStatus';
  note.setAttribute('role','status');
  note.className='studio-ik-reach-status';
  panel.append(note);

  function active(){
    return S.mode==='animation'&&tools.tool==='ik-reach';
  }
  function valid(){
    return Boolean(S.bundle&&S.scene?.puppet&&S.scene?.proof?.passed===true&&
      (S.ed.animation||[]).some(c=>c.clip_id===S.clip));
  }
  function skeleton(){
    const rig=S.scene?.puppet?.rig||{};
    return {
      joints:(rig.deform_controls||rig.controls||[]).filter(x=>x.view_id==='V'+S.view),
      edges:rig.edges||[],
      animatorControls:rig.animator_controls||[],
      constraints:S.ed.ik||[],
      viewID:'V'+S.view,
    };
  }
  function readChains(){
    if(!valid())return [];
    // Keep view-specific preview synchronized with current clip before evaluating.
    const data=skeleton();
    const candidates=reachableTwoBoneConstraints(data);
    const results=[];
    for(const chain of candidates){
      const world=[];
      for(const joint of [chain.start_id,chain.middle_id,chain.end_id]){
        const point=worldJointAnchor(joint,{
          joints:data.joints,edges:data.edges,jointPoses:S.previewPose,
        });
        if(!point||point.some(x=>!Number.isFinite(x)))break;
        world.push(point);
      }
      if(world.length!==3)continue;
      if(dist(world[0],world[1])<1e-4||dist(world[1],world[2])<1e-4)continue;
      // Scale is not an admissible length-preserving IK authoring basis.
      const scaleOk=[chain.start_id,chain.middle_id].every(id=>{
        const v=normalizePose(S.previewPose?.get(id)).scale_xy;
        return Math.abs(v[0]-1)<1e-3&&Math.abs(v[1]-1)<1e-3;
      });
      if(!scaleOk)continue;
      results.push({...chain,start:world[0],middle:world[1],end:world[2]});
    }
    return results;
  }
  function resize(){
    const v=viewport.getBoundingClientRect(),p=panel.getBoundingClientRect();
    const dpr=Math.min(devicePixelRatio||1,1.5);
    const w=Math.max(1,Math.round(v.width*dpr)),h=Math.max(1,Math.round(v.height*dpr));
    const left=v.left-p.left,top=v.top-p.top;
    if(w!==lastCanvasWidth||h!==lastCanvasHeight){
      layer.width=w;layer.height=h;ctx.setTransform(dpr,0,0,dpr,0,0);
      lastCanvasWidth=w;lastCanvasHeight=h;
    }
    if(left!==lastCanvasLeft||top!==lastCanvasTop){
      layer.style.left=left+'px';layer.style.top=top+'px';
      lastCanvasLeft=left;lastCanvasTop=top;
    }
    if(layer.style.width!==v.width+'px')layer.style.width=v.width+'px';
    if(layer.style.height!==v.height+'px')layer.style.height=v.height+'px';
  }
  function screen(world){
    const b=viewport.getBoundingClientRect();
    return [
      (world[0]*S.cam.s+S.cam.x)*b.width/Math.max(1,viewport.width),
      (world[1]*S.cam.s+S.cam.y)*b.height/Math.max(1,viewport.height),
    ];
  }
  function world(event){
    const b=viewport.getBoundingClientRect();
    const x=(event.clientX-b.left)*viewport.width/Math.max(1,b.width);
    const y=(event.clientY-b.top)*viewport.height/Math.max(1,b.height);
    return [(x-S.cam.x)/S.cam.s,(y-S.cam.y)/S.cam.s];
  }
  function showMessage(msg){
    if(msg!==previousMessage)note.textContent=previousMessage=msg;
  }
  function refresh(now){
    if(!active()){
      layer.style.pointerEvents='none';
      layer.hidden=true;note.hidden=true;
      return;
    }
    layer.hidden=false;note.hidden=false;
    layer.style.pointerEvents=valid()?'auto':'none';
    if(!valid()){
      showMessage('IK REACH / Open a qualified puppet and select a USER CLIP.');
      ctx.clearRect(0,0,layer.width,layer.height);return;
    }
    cache=readChains();
    if(!cache.length){
      layer.style.pointerEvents='none';
      showMessage('IK REACH / Create a valid two-bone IK in PUPPET; START and elbow need unique animator controls.');
    }else {
      showMessage(drag?('IK REACH / TARGET '+Math.round(drag.target[0])+', '+Math.round(drag.target[1])+
        (drag.unreachable?' · OUT OF REACH':'')+' · USER KEY PREVIEW'):
        'IK REACH / Drag the diamond on a qualified end joint. Mechanics requalification is required.');
    }
    resize();
    ctx.clearRect(0,0,layer.width,layer.height);
    const chains=drag?[drag.chain]:cache;
    for(const chain of chains){
      const [a,b,c]=[chain.start,chain.middle,drag&&chain.constraint_id===drag.chain.constraint_id?drag.target:chain.end].map(screen);
      ctx.save();
      ctx.lineWidth=1.6;ctx.setLineDash([5,6]);
      ctx.strokeStyle='rgba(176,166,255,.74)';
      ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.lineTo(...c);ctx.stroke();
      ctx.setLineDash([]);
      for(const [x,y,r] of [[...a,4],[...b,5]]){
        ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);
        ctx.fillStyle='rgba(11,17,41,.97)';ctx.fill();ctx.strokeStyle='#abb9ff';ctx.stroke();
      }
      ctx.translate(c[0],c[1]);ctx.rotate(Math.PI/4);
      ctx.beginPath();ctx.rect(-8,-8,16,16);
      ctx.fillStyle='rgba(102,81,205,.72)';
      ctx.strokeStyle='#dce5ff';ctx.lineWidth=1.6;ctx.fill();ctx.stroke();
      ctx.restore();
    }
    if(drag){
      const p=screen(drag.target);
      ctx.beginPath();ctx.arc(p[0],p[1],24,0,Math.PI*2);
      ctx.setLineDash([4,5]);
      ctx.strokeStyle=drag.unreachable?'rgba(249,181,116,.66)':'rgba(176,153,255,.35)';
      ctx.stroke();ctx.setLineDash([]);
    }
  }
  function nearest(point){
    let best=null,threshold=17;
    for(const c of cache){
      const delta=dist(screen(c.end),point);
      if(delta<threshold){threshold=delta;best=c}
    }
    return best;
  }
  function rebind(){
    if(timeline.animationEdits!==S.ed.animation)timeline.bind(S.ed.animation);
    if(timeline.clipID!==S.clip)timeline.selectClip(S.clip);
  }
  function start(event){
    if(!active()||!valid()||event.button!==0)return;
    const bounds=layer.getBoundingClientRect();
    const c=nearest([event.clientX-bounds.left,event.clientY-bounds.top]);
    if(!c)return;
    event.preventDefault();event.stopPropagation();
    rebind();
    drag={
      chain:c,before:studio.document.snapshot(),
      startPose:normalizePose(timeline.sample(c.start_control_id,S.t)),
      middlePose:normalizePose(timeline.sample(c.middle_control_id,S.t)),
      pointerID:event.pointerId,target:c.end,
      unreachable:false,changed:false,
    };
    try{layer.setPointerCapture(event.pointerId)}catch{}
  }
  function move(event){
    if(!drag||event.pointerId!==drag.pointerID)return;
    event.preventDefault();event.stopPropagation();
    const desired=world(event);
    const c=drag.chain;
    let solved;
    try{
      solved=solveTwoBoneReach({
        start:c.start,middle:c.middle,end:c.end,target:desired,
        bendDirection:c.bend_direction,mix:c.mix,
      });
    }catch(error){showMessage('IK REACH / '+error.message);return}
    const snapshot=drag.before;
    studio.document.replace(snapshot);
    rebind();
    const startPose={
      ...drag.startPose,rotation_degrees:drag.startPose.rotation_degrees+solved.start_delta_degrees,
    };
    const midPose={
      ...drag.middlePose,rotation_degrees:drag.middlePose.rotation_degrees+solved.middle_delta_degrees,
    };
    const keys=[
      [c.start_control_id,startPose],
      [c.middle_control_id,midPose],
    ];
    for(const [id,pose] of keys){
      const track=timeline.clip()?.tracks?.find(row=>(row.control_id||row.bone_id)===id);
      const existing=track?.keys?.find(key=>Math.abs(Number(key.time_seconds)-S.t)<1e-6);
      const curve=['step','linear','bezier'].includes(existing?.curve)?existing.curve:'linear';
      timeline.upsertTransformKey(id,{...pose,bezier:existing?.bezier},{time:S.t,curve});
    }
    studio.dirty.markDirty();
    studio.refreshAnimationPreview(S.t);
    drag.target=desired;
    drag.unreachable=solved.unreachable;
    drag.changed=true;
    refresh(performance.now());
  }
  function complete(event){
    if(!drag||event.pointerId!==drag.pointerID)return;
    const done=drag;drag=null;
    try{layer.releasePointerCapture(event.pointerId)}catch{}
    if(done.changed){
      studio.journal.commit('IK reach · two-bone user keyframes',done.before,studio.document.snapshot());
      studio.dirty.markDirty();
      clearTimeout(saveTimer);
      saveTimer=setTimeout(()=>{
        if(studio.dirty.status==='dirty'&&!studio.dirty.lastError)$('#saveLayer')?.click();
      },1200);
    }
    refresh(performance.now());
  }
  function cancel(){
    if(!drag)return;
    try{layer.releasePointerCapture(drag.pointerID)}catch{}
    studio.document.replace(drag.before);
    rebind();studio.refreshAnimationPreview(S.t);
    drag=null;refresh(performance.now());
  }
  function wheel(event){
    if(!active())return;
    event.preventDefault();
    const before=world(event);
    S.cam.s=clamp(S.cam.s*(event.deltaY>0?.9:1.1),.05,20);
    const rect=viewport.getBoundingClientRect();
    const intrinsic=[(event.clientX-rect.left)*viewport.width/rect.width,(event.clientY-rect.top)*viewport.height/rect.height];
    S.cam.x=intrinsic[0]-before[0]*S.cam.s;
    S.cam.y=intrinsic[1]-before[1]*S.cam.s;
    refresh(performance.now());
  }
  layer.addEventListener('pointerdown',start);
  layer.addEventListener('pointermove',move);
  layer.addEventListener('pointerup',complete);
  layer.addEventListener('pointercancel',cancel);
  layer.addEventListener('wheel',wheel,{passive:false});
  addEventListener('keydown',e=>{
    if(e.key==='Escape')cancel();
    if(e.key.toLowerCase()==='i'&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&
       !(e.target instanceof HTMLInputElement)&&!(e.target instanceof HTMLTextAreaElement)&&
       !(e.target instanceof HTMLSelectElement)&&!e.target?.isContentEditable&&
       S.mode==='animation')tools.setTool('ik-reach');
  });
  addEventListener('resize',()=>refresh(performance.now()),{passive:true});
  tools.addEventListener('change',()=>refresh(performance.now()));
  const ro=new ResizeObserver(()=>refresh(performance.now()));ro.observe(viewport);
  function tick(now){
    if(!document.hidden&&now-lastDraw>140){
      lastDraw=now;
      if(active())refresh(now);
      else if(!layer.hidden)refresh(now);
    }
    requestAnimationFrame(tick);
  }
  refresh(performance.now());
  requestAnimationFrame(tick);
}
boot();
