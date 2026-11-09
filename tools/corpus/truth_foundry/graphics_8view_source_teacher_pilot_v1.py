#!/usr/bin/env python3
"""Private eight-view source-conditioned 3D raster pilot, NOT approved 2D art.

Same SHA-checked canonical GLB rendered in Blender from eight fixed azimuths,
plus a clearly designated 2D stylization candidate. Morphological front axis,
authored original texture rights, source mesh parity, 2D art review and user
single-view inference all remain UNKNOWN / NOT_RUN. Do not train on this pilot.

Run --select (host Python), --render (Blender Python), --postprocess (host Python).
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from collections import Counter
from pathlib import Path

CATS=('serpentine','marine','insectoid','avian','quadrupedal','articulated_rigid')
SCHEMA='RealSaS.GraphicsEightViewSourceTeacherPilot.v1'
ANGLES=tuple(range(0,360,45))


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def choose_sources(manifest:Path,out:Path):
 with Path(manifest).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 if len(rows)!=96:raise ValueError('EXPECTED_96_PINNED_SOURCE_LIST')
 picks=[]
 for cat in CATS:
  cands=[x for x in rows if x['category_provider_claim']==cat and
         x['download_status']=='DOWNLOADED' and x['glb_valid']=='True' and
         int(x['glb_image_count'])>0 and int(x['glb_texture_count'])>0 and
         int(x['glb_triangle_indices_count'])>100 and x['download_sha256']]
  if not cands:raise ValueError('MISSING_TEXTURED_RARE_MORPHOLOGY:'+cat)
  cands.sort(key=lambda x:(int(x['glb_triangle_indices_count']),x['object_type']))
  r=cands[0]
  picks.append({'object_type':r['object_type'],'morphology_provider_claim':cat,
                'glb_sha256':r['download_sha256'],'bytes':r['downloaded_bytes'],
                'source_original_rights':'UNKNOWN',
                'anatomical_front_axis':'UNKNOWN',
                'student_single_view_input_not_produced':'True',
                'source_original_parity':'UNKNOWN','training_authorized':'False'})
 out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(picks[0]));w.writeheader();w.writerows(picks)
 return {'selected':len(picks),'morphologies':list(CATS),'selection_sha256':sha(out)}


def parse_selection(path):
 with Path(path).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 if len(rows)!=6 or len({r['morphology_provider_claim'] for r in rows})!=6:
  raise ValueError('EXPECTED_SIX_RARE_CATEGORIES')
 for r in rows:
  if not r['object_type'].replace('_','').isalnum() or len(r['glb_sha256'])!=64 or r['training_authorized']!='False':
   raise ValueError('UNSAFE_OR_UNPINNED_SOURCE')
 return rows


def render_bpy(selection:Path,glb_dir:Path,out:Path):
 import bpy
 from mathutils import Vector
 rows=parse_selection(selection)
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 receipts=[];problems=[]
 for r in rows:
  name=r['object_type'];file=Path(glb_dir)/(name+'.glb')
  if not file.is_file() or sha(file)!=r['glb_sha256']:
   raise ValueError('CANONICAL_GLB_SHA_OR_MISSING:'+name)
  try:
   bpy.ops.wm.read_factory_settings(use_empty=True)
   bpy.ops.import_scene.gltf(filepath=str(file))
   objs=[o for o in bpy.context.scene.objects if o.type=='MESH']
   if not objs:raise ValueError('NO_MESH_IMPORTED')
   bpy.context.view_layer.update()
   corners=[]
   for o in objs:
    corners.extend([o.matrix_world @ Vector(c) for c in o.bound_box])
   low=Vector([min(c[i] for c in corners) for i in range(3)])
   high=Vector([max(c[i] for c in corners) for i in range(3)])
   center=(low+high)/2
   extent=high-low
   diameter=max(extent.length,0.1)
   # camera nomenclature is orbit-relative ONLY, not anatomical front.
   cam_data=bpy.data.cameras.new('TruthFoundryOrbitCamera')
   camera=bpy.data.objects.new('TruthFoundryOrbitCamera',cam_data)
   bpy.context.collection.objects.link(camera)
   bpy.context.scene.camera=camera
   cam_data.type='ORTHO'
   cam_data.ortho_scale=max(diameter*1.35,0.2)
   for i,loc in enumerate([(4,-6,7),(0,4,3)]):
    data=bpy.data.lights.new('TF_Light_'+str(i),'AREA')
    data.energy=(600 if i==0 else 350)
    data.shape='DISK';data.size=diameter*3
    obj=bpy.data.objects.new(data.name,data);bpy.context.collection.objects.link(obj)
    obj.location=center+Vector(loc).normalized()*diameter*2
    obj.rotation_euler=(center-obj.location).to_track_quat('-Z','Y').to_euler()
   scene=bpy.context.scene
   try:scene.render.engine='BLENDER_EEVEE_NEXT'
   except Exception:scene.render.engine='BLENDER_EEVEE'
   scene.render.resolution_x=256;scene.render.resolution_y=256
   scene.render.resolution_percentage=100
   scene.render.film_transparent=True
   scene.render.image_settings.file_format='PNG'
   scene.render.image_settings.color_mode='RGBA'
   scene.render.image_settings.color_depth='8'
   if scene.world:
    scene.world.color=(0.12,0.12,0.12)
   try:scene.view_settings.view_transform='Standard'
   except Exception:pass
   out_dir=out/name;out_dir.mkdir(parents=True,exist_ok=True)
   for angle in ANGLES:
    a=math.radians(angle)
    camera.location=center+Vector((math.sin(a),-math.cos(a),0.32)).normalized()*diameter*2.7
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    # Force the actual evaluated camera transform into the evidence receipt.
    bpy.context.view_layer.update()
    scene.render.filepath=str(out_dir/('orbit_%03d_native.png'%angle))
    bpy.ops.render.render(write_still=True)
    im=Path(scene.render.filepath)
    if not im.is_file() or im.stat().st_size<200:raise ValueError('EMPTY_NATIVE_RENDER')
    receipts.append({'object_type':name,'morphology_weak':r['morphology_provider_claim'],
      'glb_sha256':r['glb_sha256'],'camera_azimuth_deg_relative':angle,
      'camera_source_frame':'CANONICAL_GLB_BLENDER_IMPORT__ANATOMICAL_FRONT_UNKNOWN',
      'camera_matrix_world_rowmajor':[list(row) for row in camera.matrix_world],
      'camera_ortho_scale':cam_data.ortho_scale,
      'source_bbox_low':list(low),'source_bbox_high':list(high),
      'native_rgba_sha256':sha(im),'native_rgba_path':str(im.relative_to(out)),
      'style_candidate_sha256':'NOT_BUILT','source_pixel_original_authority':'NOT_VERIFIED',
      'source_rights_court':'NOT_RUN','style_art_court':'NOT_RUN',
      'source_part_semantic_court':'NOT_RUN','teacher_student_leakage_court':'NOT_RUN',
      'training_authorized':False})
  except Exception as exc:
   problems.append({'object_type':name,'error_type':type(exc).__name__,'message':str(exc)[:240]})
 with (out/'RENDER_RECEIPTS_V1.jsonl').open('w',encoding='utf-8') as f:
  for receipt in receipts:f.write(json.dumps(receipt,sort_keys=True)+'\n')
 (out/'RENDER_FAILURES_V1.json').write_text(json.dumps(problems,indent=2)+'\n')
 return {'rendered':len(receipts),'errors':problems}


def stylize_postprocess(out:Path):
 from PIL import Image,ImageFilter,ImageOps
 out=Path(out)
 receipts=out/'RENDER_RECEIPTS_V1.jsonl'
 records=[json.loads(line) for line in receipts.read_text().splitlines() if line.strip()]
 rgba_valid=0
 for r in records:
  image_path=out/r['native_rgba_path']
  with Image.open(image_path) as original:
   rgba=original.convert('RGBA')
   rgba.load()
   alpha=rgba.getchannel('A')
   if alpha.getbbox() is None:
    raise ValueError('EMPTY_ALPHA_SILHOUETTE:'+str(image_path))
   # Pilot only: indexed palette image-space stylization, not a trained/approved 2D artist shader.
   rgb=rgba.convert('RGB')
   small=rgb.quantize(colors=24,method=Image.Quantize.FASTOCTREE).convert('RGB')
   edge=alpha.filter(ImageFilter.FIND_EDGES)
   outline=ImageOps.invert(edge)
   outim=small.convert('RGBA');outim.putalpha(alpha)
   outim.paste((27,24,32,255),(0,0,outim.width,outim.height),ImageOps.invert(outline).point(lambda v:255 if v>160 else 0))
   outim.putalpha(alpha)
   candidate=image_path.with_name(image_path.name.replace('_native.png','_posterized_candidate.png'))
   outim.save(candidate,optimize=True)
   r['style_candidate_sha256']=sha(candidate)
   r['style_candidate_path']=str(candidate.relative_to(out))
   r['style_art_court']='PROCEDURAL_2D_CANDIDATE__NOT_APPROVED'
   rgba_valid+=1
 with receipts.open('w',encoding='utf-8') as f:
  for r in records:f.write(json.dumps(r,sort_keys=True)+'\n')
 summary={'schema':SCHEMA,'source_morphology_groups_targeted':len(CATS),
  'canonical_source_orbit_native_renders':rgba_valid,
  'source_relative_8_direction_complete_count':sum(sum(x['object_type']==name for x in records)==8 for name in {x['object_type'] for x in records}),
  'posterized_2d_art_candidates_not_approved':rgba_valid,
  'original_source_rights_verified':0,'original_source_bytes_parity_pass':0,
  'anatomical_front_camera_verified':0,'original_2d_art_approved':0,
  'single_view_student_prior_pairs_admitted':0,'independent_8k_truth_admission':0,
  'render_receipts_sha256':sha(receipts),
  'training_authorized':False}
 (out/'SUMMARY_V1.json').write_text(json.dumps(summary,indent=2)+'\n')
 return summary


def main():
 p=argparse.ArgumentParser()
 g=p.add_mutually_exclusive_group(required=True)
 g.add_argument('--select',action='store_true')
 g.add_argument('--render',action='store_true')
 g.add_argument('--postprocess',action='store_true')
 p.add_argument('--manifest',type=Path)
 p.add_argument('--selection',type=Path)
 p.add_argument('--glb-dir',type=Path)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.select:
  print(json.dumps(choose_sources(a.manifest,a.out),indent=2))
 elif a.render:
  print(json.dumps(render_bpy(a.selection,a.glb_dir,a.out),indent=2))
 else:
  print(json.dumps(stylize_postprocess(a.out),indent=2))

if __name__=='__main__':
 if '--' in sys.argv:
  # Blender passes script arguments after `--`; argv[0] must remain the
  # program name, otherwise argparse consumes the first required flag.
  sys.argv=[sys.argv[0]]+sys.argv[sys.argv.index('--')+1:]
 main()