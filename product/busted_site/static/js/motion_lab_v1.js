/* Motion Lab plays qualification-owned runtime clips, never synthetic presets. */
(() => {
  'use strict';
  const $=s=>document.querySelector(s), url=new URL(location.href);
  let bundle='',scene=null,clips=[],clip=null,sourceImage=null,sourceUrl=null,playing=false,busy=false,lastPaint=0,started=0,clock=0,requestGeneration=0;
  const canvas=$('#motionCanvas'),ctx=canvas.getContext('2d');
  const labels=['Idle','Run','Walk','Attack','Jump','Roll','Death','Other'];
  const message=t=>{$('#pageMessage').textContent=t};
  function nav(){
    for(const a of document.querySelectorAll('[data-nav]')){
      const u=new URL(a.getAttribute('href').split('?')[0],location.href);
      if(bundle)u.searchParams.set('bundle',bundle);
      a.href=u.href;
    }
    const u=new URL(location.href);
    if(bundle)u.searchParams.set('bundle',bundle);else u.searchParams.delete('bundle');
    history.replaceState(null,'',u.href);
  }
  const n=v=>Number.isFinite(Number(v))?Number(v):0;
  function classify(c){
    const s=(String(c.intent||'')+' '+String(c.clip_id||'')).toLowerCase();
    return labels.find(k=>k!=='Other'&&s.includes(k.toLowerCase()))||'Other';
  }
  async function json(path){
    const response=await fetch(path,{cache:'no-store'});
    const payload=await response.json().catch(()=>({}));
    if(!response.ok)throw Error(payload?.message||payload?.error||'Backend not connected ('+response.status+')');
    return payload;
  }
  function reset(){
    requestGeneration++;playing=false;busy=false;clock=0;clip=null;clips=[];scene=null;
    $('#motionEmpty').hidden=false;$('#proofStatus').textContent='NO QUALIFIED PUPPET';$('#proofStatus').classList.remove('qualified');
    $('#clipLabel').textContent='NO CLIP SELECTED';$('#detailClip').textContent='—';$('#detailIntent').textContent='—';
    $('#detailDuration').textContent='—';$('#detailProof').textContent='NOT LOADED';
    $('#motionSeek').disabled=true;$('#motionPlay').disabled=true;$('#motionStop').disabled=true;$('#motionPlay').textContent='▶ PLAY';
    $('#clipCount').textContent='0 CLIPS';$('#clipList').replaceChildren();$('#motionCanvas').getContext('2d').clearRect(0,0,canvas.width,canvas.height);
    if(sourceUrl)URL.revokeObjectURL(sourceUrl);sourceUrl=null;sourceImage=null;
    buildShelf();
  }
  function buildShelf(){
    const shelf=$('#presetShelf');shelf.replaceChildren();
    labels.forEach(label=>{
      const matching=clips.filter(c=>classify(c)===label);
      const button=document.createElement('button');button.type='button';button.disabled=!matching.length;
      const heading=document.createElement('span');heading.textContent=label;
      const note=document.createElement('small');note.textContent=matching.length?matching.length+' BAKED':'UNAVAILABLE';
      button.append(heading,note);button.classList.toggle('ready',!!matching.length);
      button.addEventListener('click',()=>selectClip(matching[0]));
      shelf.appendChild(button);
    });
  }
  function updateClips(){
    buildShelf();const list=$('#clipList');list.replaceChildren();
    $('#clipCount').textContent=clips.length+' CLIPS';
    if(!clips.length){const p=document.createElement('p');p.className='soft-note';p.textContent='No qualified motion clips are attached to this puppet.';list.append(p);return}
    clips.forEach(row=>{
      const button=document.createElement('button');button.type='button';button.classList.toggle('selected',clip?.clip_id===row.clip_id);
      const name=document.createElement('span');name.textContent=String(row.clip_id);
      const kind=document.createElement('small');kind.textContent=classify(row);
      button.append(name,kind);button.addEventListener('click',()=>selectClip(row));list.append(button);
    });
  }
  async function image(){
    const source=scene?.index?.source_views?.find(x=>Number(x.view_index)===0);
    const path=source?.assets?.clean_cutout?.path||source?.assets?.raw_image?.path;
    if(!path)return;
    try{
      const r=await fetch('/api/blob?bundle='+encodeURIComponent(bundle)+'&path='+encodeURIComponent(path));
      if(!r.ok)return;
      sourceUrl=URL.createObjectURL(await r.blob());const img=new Image();
      await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=reject;img.src=sourceUrl});
      sourceImage=img;
    }catch{sourceImage=null}
  }
  function triangle(dst,src){
    const [a,b,c]=src,[A,B,C]=dst;
    const det=a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]);
    if(Math.abs(det)<1e-8)return;
    const coeff=(values,axis)=>{
      const [v0,v1,v2]=values;
      return [
       (v0[axis]*(b[1]-c[1])+v1[axis]*(c[1]-a[1])+v2[axis]*(a[1]-b[1]))/det,
       (v0[axis]*(c[0]-b[0])+v1[axis]*(a[0]-c[0])+v2[axis]*(b[0]-a[0]))/det,
       (v0[axis]*(b[0]*c[1]-c[0]*b[1])+v1[axis]*(c[0]*a[1]-a[0]*c[1])+v2[axis]*(a[0]*b[1]-b[0]*a[1]))/det
      ];
    };
    const x=coeff([A,B,C],0),y=coeff([A,B,C],1);
    ctx.save();ctx.beginPath();ctx.moveTo(...A);ctx.lineTo(...B);ctx.lineTo(...C);ctx.closePath();ctx.clip();
    ctx.transform(x[0],y[0],x[1],y[1],x[2],y[2]);ctx.drawImage(sourceImage,0,0);ctx.restore();
  }
  function draw(payload){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    const meshes=(scene?.puppet?.meshes||[]).filter(m=>m.view_id==='V0');
    const frames=payload?.frame?.mesh_vertices_by_id||{};
    const vertices=meshes.flatMap(m=>frames[m.mesh_id]||m.vertices||[]).filter(v=>Array.isArray(v)&&v.length>=2);
    if(!vertices.length)return;
    const xs=vertices.map(x=>n(x[0])),ys=vertices.map(x=>n(x[1]));
    const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
    const scale=Math.min((canvas.width-110)/Math.max(1,x1-x0),(canvas.height-110)/Math.max(1,y1-y0));
    const ox=canvas.width/2-(x0+x1)*.5*scale,oy=canvas.height/2-(y0+y1)*.5*scale;
    const fit=p=>[n(p[0])*scale+ox,n(p[1])*scale+oy];
    const order=payload?.frame?.render_order_by_view?.V0||[];
    const sorted=[...meshes].sort((a,b)=>order.indexOf(a.mesh_id)-order.indexOf(b.mesh_id));
    for(const m of sorted){
      const frameVerts=frames[m.mesh_id]||m.vertices||[];
      for(const ids of m.triangles||[]){
        const dst=ids.map(i=>fit(frameVerts[i])),src=ids.map(i=>m.vertices[i]);
        if(dst.some(p=>!p)||src.some(p=>!p))continue;
        if(sourceImage)triangle(dst,src);
        else{ctx.beginPath();ctx.moveTo(...dst[0]);ctx.lineTo(...dst[1]);ctx.lineTo(...dst[2]);ctx.closePath();ctx.fillStyle='rgba(122,207,242,.24)';ctx.fill();ctx.strokeStyle='rgba(177,231,255,.34)';ctx.stroke()}
      }
    }
  }
  async function sample(t){
    if(!clip||busy||document.hidden)return;
    busy=true;const generation=requestGeneration;
    try{
      const frame=await json('/api/runtime/frame?bundle='+encodeURIComponent(bundle)+'&clip='+encodeURIComponent(clip.clip_id)+'&time='+encodeURIComponent(t));
      if(generation===requestGeneration){draw(frame);const duration=Math.max(.001,n(clip.duration_seconds)||1);$('#motionSeek').value=String(Math.round(t/duration*1000));$('#motionTime').textContent=t.toFixed(2)+' / '+duration.toFixed(2)}
    }catch(e){playing=false;message('PREVIEW ERROR · '+e.message)}
    finally{busy=false}
  }
  function selectClip(row){
    clip=row;clock=0;playing=false;requestGeneration++;
    $('#motionEmpty').hidden=true;$('#clipLabel').textContent=String(row.clip_id);
    $('#detailClip').textContent=String(row.clip_id);$('#detailIntent').textContent=String(row.intent||classify(row));
    $('#detailDuration').textContent=(n(row.duration_seconds)).toFixed(2)+' SECONDS';$('#detailProof').textContent='QUALIFIED RUNTIME';
    $('#motionSeek').disabled=false;$('#motionPlay').disabled=false;$('#motionStop').disabled=false;$('#motionPlay').textContent='▶ PLAY';
    updateClips();void sample(0);
  }
  async function load(){
    reset();
    bundle=$('#bundleInput').value.trim();
    nav();
    if(!bundle){message('ENTER A QUALIFIED PUPPET BUNDLE');return}
    message('VERIFYING PRODUCT PROOF…');
    try{
      scene=await json('/api/living/scene?bundle='+encodeURIComponent(bundle));
      if(scene?.proof?.passed!==true)throw Error('Bundle has no qualified ProductProof');
      const data=await json('/api/runtime/clips?bundle='+encodeURIComponent(bundle));
      clips=(Array.isArray(data.clips)?data.clips:[]).filter(c=>c&&c.clip_id&&n(c.duration_seconds)>0);
      $('#proofStatus').textContent='PRODUCT PROOF · PASS';$('#proofStatus').classList.add('qualified');
      await image();updateClips();
      if(clips.length)selectClip(clips[0]);
      message(clips.length?'QUALIFIED PUPPET LOADED · '+clips.length+' BAKED CLIPS':'QUALIFIED PUPPET LOADED · NO BAKED CLIPS');
    }catch(e){reset();message('LOAD UNAVAILABLE · '+e.message)}
  }
  function animate(now){
    if(playing&&clip&&!document.hidden){
      const dt=Math.min(.15,(now-started)/1000);started=now;
      const duration=Math.max(.001,n(clip.duration_seconds));
      clock=(clock+Math.max(0,dt))%duration;
      if(now-lastPaint>95&&!busy){lastPaint=now;void sample(clock)}
    }else started=now;
    requestAnimationFrame(animate);
  }
  $('#loadBundle').addEventListener('click',load);
  $('#bundleInput').addEventListener('keydown',e=>{if(e.key==='Enter')void load()});
  $('#motionPlay').addEventListener('click',()=>{
    if(!clip)return;playing=!playing;started=performance.now();$('#motionPlay').textContent=playing?'❚❚ PAUSE':'▶ PLAY';
  });
  $('#motionStop').addEventListener('click',()=>{playing=false;clock=0;$('#motionPlay').textContent='▶ PLAY';void sample(0)});
  $('#motionSeek').addEventListener('input',e=>{if(!clip)return;playing=false;$('#motionPlay').textContent='▶ PLAY';clock=n(e.target.value)/1000*n(clip.duration_seconds);void sample(clock)});
  reset();$('#bundleInput').value=new URLSearchParams(location.search).get('bundle')||'';
  if($('#bundleInput').value)void load();
  requestAnimationFrame(animate);
})();
