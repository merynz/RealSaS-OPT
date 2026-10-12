#!/usr/bin/env python3
"""Export an offline read-only investor inspector from exact M/G/W frame data."""
from __future__ import annotations
import argparse,base64,hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.demo.knight_preview_media_v1 import save_png_checked
ROOT=Path(__file__).resolve().parents[2]
CLIPS=[('demo_idle_v1','Idle'),('demo_run_v1','Run'),('demo_slash_v1','Slash'),('demo_jump_v1','Jump'),('demo_death_v1','Death'),('demo_roll_sword_v1','Sword roll'),('demo_jump_attack_v1','Jump attack')]

def packed(array,dtype):return base64.b64encode(np.asarray(array,dtype=dtype).tobytes()).decode('ascii')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(renders,out,allow_partial=False):
 out.mkdir(parents=True,exist_ok=True);assets=out/'demo';assets.mkdir(exist_ok=True);movies=out/'videos';movies.mkdir(exist_ok=True)
 result=[];common=None;reports=[];integrity={}
 for cid,title in CLIPS:
  folder=renders/cid;report_path=folder/'source_motion_evidence.json'
  if not report_path.exists():
   if allow_partial:continue
   raise RuntimeError('MOTION_RENDER_MISSING:'+cid)
  report=json.loads(report_path.read_text());prefix=report['clip_kind'].lower();pose_path=folder/f'compiled_{prefix}_m_poses.npz'
  if not pose_path.exists():raise RuntimeError('FRAME_WITNESS_MISSING:'+cid)
  with np.load(pose_path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
  expected={k:hashlib.sha256(data[k].tobytes()).hexdigest() for k in ['rest','faces','weights','axis','parents']}
  if common is None:common=data
  elif any(not np.array_equal(common[k],data[k]) for k in ['rest','faces','weights','axis','parents','attachment_owners','vertex_part','face_part']):raise RuntimeError('CHARACTER_DOMAIN_DRIFT:'+cid)
  if report['preview_fps']!=30 or report['preview_duration_seconds']<4:raise RuntimeError('INVESTOR_TIMING_CONTRACT:'+cid)
  res=int(report['per_view_resolution']);n=len(data['times']);columns=8;rows=(n+columns-1)//columns;sprites=[]
  if not all((folder/f'{prefix}_v{v}_{i:03d}.png').exists() for v in range(8) for i in range(n)):
   if allow_partial:continue
   raise RuntimeError('EIGHT_VIEW_FRAMES_INCOMPLETE:'+cid)
  for v in range(8):
   sprite=Image.new('RGBA',(res*columns,res*rows),(0,0,0,0))
   for i in range(n):
    with Image.open(folder/f'{prefix}_v{v}_{i:03d}.png') as image:sprite.paste(image.convert('RGBA'),((i%columns)*res,(i//columns)*res))
   dest=assets/f'{prefix}_v{v}.png';save_png_checked(sprite,dest);sprites.append(f'demo/{dest.name}');integrity[str(dest.relative_to(out))]=digest(dest)
  refs=sorted(set([0,n//2,n-1]));tracks=json.loads((folder/f'compiled_{prefix}_tracks.json').read_text())
  if 'mapping' not in locals():mapping=tracks['target_to_source']
  result.append(dict(id=cid,title=title,cameras=report['output_cameras'],kind=report['clip_kind'],loop=report['source_loop'],fps=30,previewCount=len(data['playback_times']),sampleCount=n,columns=columns,sprites=sprites,matrices=packed(data['skin_matrices'],'<f8'),attach=packed(data['attachment_matrices'],'<f8'),indices=packed(data['playback_frame_indices'],'<u4'),ranks=packed(data['part_orders'],'<f4'),overlap=report['overlap_policy'],referenceFrames=refs,reference=packed(data['poses'][refs],'<f8')))
  reports.append(report)
  for video in ([] if allow_partial else sorted(p for p in folder.glob('Knight_*.mp4') if '.writing.' not in p.name)):
   probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate,nb_frames,duration','-of','json',str(video)],text=True))['streams'][0]
   if probe['r_frame_rate']!='30/1' or float(probe['duration'])<4 or int(probe['nb_frames'])<120:raise RuntimeError('MP4_DELIVERY_CONTRACT:'+video.name)
   dest=movies/video.name;shutil.copy2(video,dest);integrity[str(dest.relative_to(out))]=digest(dest)
  for gif in folder.glob('Knight_*V0-V7.gif'):shutil.copy2(gif,movies/gif.name)
  print('EXPORTED',title,n,'source poses',len(data['playback_times']),'preview frames',flush=True)
 if common is None or not result:raise RuntimeError('NO_COMPLETE_RENDERED_CHARACTER')
 names={'Bone':'Root','Body':'Body root','Abdomen':'Abdomen','Torso':'Chest','Head':'Head','UpperArm.L':'Left upper arm','UpperArm.R':'Right upper arm','LowerArm.L':'Left forearm','LowerArm.R':'Right forearm','Palm.L':'Left wrist','Palm.R':'Right wrist','MiddleHand.L':'Left hand','MiddleHand.R':'Right hand','UpperLeg.L':'Left thigh','UpperLeg.R':'Right thigh','LowerLeg.L':'Left shin','LowerLeg.R':'Right shin'}
 labels=[]
 for i in range(len(common['parents'])):
  sid=mapping.get(str(i));labels.append(names.get(sid,sid if sid else f'Control {i:02d}'))
 for binding in reports[0]['attachment_bindings']:labels[binding['slot']]='Sword attachment' if binding['owner']==1 else 'Shield attachment'
 baseline=json.loads((ROOT/'tests/fixtures/single_m_caa_knight_v1/baseline.json').read_text())
 payload=dict(schema='RealSaS.OfflineKnightInspector.v1',resolution=int(reports[0]['per_view_resolution']),rest=packed(common['rest'],'<f8'),faces=packed(common['faces'],'<u4'),weights=packed(common['weights'],'<f8'),axis=packed(common['axis'],'<f8'),parents=packed(common['parents'],'<i4'),owners=packed(common['attachment_owners'],'<i4'),facePart=packed(common['face_part'],'<i4'),partCount=int(common['face_part'].max()+1),attachOwners=[r['owner'] for r in reports[0]['attachment_bindings']],boneLabels=labels,cameras=baseline['cameras'],viewLabels=['front','front diagonal','side','rear diagonal','back','rear diagonal','side','front diagonal'],clips=result)
 (assets/'data.js').write_text('window.REALSAS_DEMO='+json.dumps(payload,separators=(',',':'))+';\n')
 template=ROOT/'tools/demo/templates/knight_investor_v1.html';shutil.copy2(template,out/'START_DEMO.html')
 verification=out/'verification';verification.mkdir(exist_ok=True)
 (verification/'frame_and_render_evidence.json').write_text(json.dumps(reports,indent=2)+'\n')
 info=dict(schema='RealSaS.InvestorDemoManifest.v1',clip_count=len(result),mesh_sha256=expected['rest'],topology_sha256=expected['faces'],weights_sha256=expected['weights'],axis_sha256=expected['axis'],hierarchy_sha256=expected['parents'],fps=30,minimum_duration_seconds=4,directions=8,live_browser_skinning=True,live_browser_rigid_surface_attachments=True,canonical_weights_fixed=True,product_qualification_claimed=False,semantic_overlap_qualified=False,files=integrity)
 info['implementation_source_sha256']={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'compiler/realsas_compiler_core/motion_retarget_frames_v3.py',ROOT/'compiler/realsas_compiler_core/rigid_attachment_surface_anchor_v1.py',ROOT/'compiler/realsas_compiler_core/presentation_part_order_v1.py',ROOT/'compiler/realsas_compiler_core/single_m_caa_v1.py',ROOT/'tools/demo/render_knight_source_motion_v3.py',template]}
 (verification/'package_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
 (out/'README.txt').write_text('RealSaS - Knight Interactive Demo\n\n1. Extract the entire ZIP.\n2. Double-click START_DEMO.html in Chrome, Edge or Firefox.\n3. Select an animation and camera direction. Press Space to pause.\n4. Reveal the structure, select a bone, and scrub the same frame across artwork, rig, mesh and weights.\n\nNo installation, account, network connection or backend is required.\nThe videos folder contains all 7 motions in 8 directions at 30 FPS, with durations chosen for readable playback.\nNon-looping motions hold their terminal pose; source motion speed is preserved.\n\nThis technical preview uses one exact Knight mesh, hierarchy and skin field.\nThe browser computes live skinning from those weights and the actual frame matrices.\nRigid equipment consumes compiled surface-addressed transforms. Source-owned artwork is pre-rendered from the matching frame.\nSkin weight values stay fixed; their colors travel with the posed mesh. Live deformation contribution is a separate view.\n\nGrip and artistic overlap are improved engineering candidates, not final product qualification.\nThe verification folder contains exact identities and measured limits.\nSource character and native motion: Quaternius Animated Knight Pack (CC0).\nhttps://quaternius.com/packs/knightcharacter.html\n',encoding='utf8')
 print('READY',out,flush=True)
 return out

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--renders',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args();build(args.renders.resolve(),args.output.resolve(),args.allow_partial)
