/* Truthful native export reference: no fake file download or unsupported format promises. */
import {apiFetch} from './api_gateway_v1.js';
(() => {
  'use strict';
  const $=s=>document.querySelector(s);
  let bundle='',qualified=false;
  const message=t=>$('#pageMessage').textContent=t;
  async function json(path,options){
    const r=await apiFetch(path,{cache:'no-store',...options});
    const payload=await r.json().catch(()=>({}));
    if(!r.ok)throw Error(payload?.message||payload?.error||'Backend not connected ('+r.status+')');
    return payload;
  }
  function nav(){
    for(const a of document.querySelectorAll('[data-nav]')){
      const u=new URL(a.getAttribute('href').split('?')[0],location.href);
      if(bundle)u.searchParams.set('bundle',bundle);
      a.href=u.href;
    }
    const u=new URL(location.href);if(bundle)u.searchParams.set('bundle',bundle);else u.searchParams.delete('bundle');
    history.replaceState(null,'',u.href);
  }
  function reset(){
    qualified=false;$('#prepareExport').disabled=true;$('#exportProof').textContent='NOT VERIFIED';
    $('#exportPackage').textContent='NOT CHECKED';$('#proofStatus').textContent='NO QUALIFIED PUPPET';$('#proofStatus').classList.remove('qualified');
    $('#exportResult').textContent='Open a qualified character to inspect its export.';$('#exportResult').classList.remove('success');
  }
  async function load(){
    reset();bundle=$('#bundleInput').value.trim();nav();
    if(!bundle){message('ENTER A QUALIFIED PUPPET BUNDLE');return}
    message('VERIFYING PRODUCT PROOF…');
    try{
      const scene=await json('/api/living/scene?bundle='+encodeURIComponent(bundle));
      if(scene?.proof?.passed!==true)throw Error('Bundle has no qualified ProductProof');
      qualified=true;$('#prepareExport').disabled=false;$('#exportProof').textContent='VERIFIED PASS';
      $('#proofStatus').textContent='PRODUCT PROOF · PASS';$('#proofStatus').classList.add('qualified');
      $('#exportResult').textContent='Proof-qualified puppet ready. Verify whether the native .rss package exists on the server.';
      message('READY · VERIFIED PRODUCT PROOF');
    }catch(e){reset();message('EXPORT BLOCKED · '+e.message)}
  }
  $('#loadBundle').addEventListener('click',load);
  $('#bundleInput').addEventListener('keydown',e=>{if(e.key==='Enter')void load()});
  $('#prepareExport').addEventListener('click',async()=>{
    if(!qualified)return;
    $('#prepareExport').disabled=true;$('#exportResult').textContent='Checking native runtime package…';
    try{
      const result=await json('/api/export',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({bundle})});
      if(result.authority!=='EXISTING_PROOF_GATED_NATIVE_RUNTIME'||!String(result.package_path||'').endsWith('.rss'))throw Error('Server returned no verified native package');
      $('#exportPackage').textContent='VERIFIED .RSS';
      $('#exportResult').classList.add('success');
      $('#exportResult').textContent='Native .rss package verified on the product server. Direct browser download is not yet connected; no file was downloaded.';
      message('NATIVE PACKAGE FOUND · DELIVERY ENDPOINT NOT CONNECTED');
    }catch(e){$('#exportPackage').textContent='NOT AVAILABLE';$('#exportResult').textContent='Export unavailable: '+e.message;message('NATIVE PACKAGE NOT AVAILABLE')}
    finally{$('#prepareExport').disabled=!qualified}
  });
  reset();$('#bundleInput').value=new URLSearchParams(location.search).get('bundle')||'';
  if($('#bundleInput').value)void load();
})();
