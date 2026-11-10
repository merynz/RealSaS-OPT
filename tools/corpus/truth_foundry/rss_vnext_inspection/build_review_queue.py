#!/usr/bin/env python3
"""Review routing for complex normalized derivative subjects, not semantic decisions."""
from __future__ import annotations
import csv,json,pathlib,sys,collections

def run(root):
 rows=[json.loads(l) for l in (root/'DEEP_DERIVATIVE_SUBJECT_AUDIT_V1.jsonl').read_text().splitlines()]
 out=[]
 for r in rows:
  factors=[]
  if r.get('mesh_object_count',0)>=10:factors.append('MANY_MESH_OBJECTS')
  if r.get('connected_triangle_surface_islands',0)>=50:factors.append('MANY_DISCONNECTED_ISLANDS')
  if r.get('root_joints',0)>1:factors.append('MULTIPLE_ARMATURE_ROOTS')
  if r.get('source_object_crossing_faces',0)>0:factors.append('CROSS_OBJECT_FACES')
  if r.get('nonmanifold_edges',0)>0:factors.append('NONMANIFOLD_SOURCE_DERIVATIVE')
  if r.get('boundary_edges',0)>0:factors.append('OPEN_SURFACE')
  if r.get('skin_zero_sum_vertices',0)>0:factors.append('UNBOUND_VERTICES')
  if r.get('skin_max_positive_influences_per_vertex',0)>8:factors.append('HIGH_INFLUENCE_CARDINALITY')
  # Priority is triage only, not a character classification or negative proof.
  sev=3*('CROSS_OBJECT_FACES' in factors)+2*('MANY_MESH_OBJECTS' in factors)+2*('MANY_DISCONNECTED_ISLANDS' in factors)+2*('UNBOUND_VERTICES' in factors)+('NONMANIFOLD_SOURCE_DERIVATIVE' in factors)+('MULTIPLE_ARMATURE_ROOTS' in factors)+('OPEN_SURFACE' in factors)+('HIGH_INFLUENCE_CARDINALITY' in factors)
  out.append({'source_key':r['source_key_fragment'],'receipt_version':r['court_version'],'risk_priority_score':sev,'risk_factors':'|'.join(factors),
  'mesh_object_count':r.get('mesh_object_count',0),'topology_island_count':r.get('connected_triangle_surface_islands',0),
  'texture_material_scene_verified':'NO','native_pose_and_front_verified':'NO','morphology_semantic_verified':'NO',
  'render_art_caa_verified':'NO','verification_required':'NATIVE_SOURCE_SCENE_OPEN_AND_MULTIVIEW_RENDER_REVIEW',
  'human_visual_review_status':'PENDING','derivative_structure_sha256':r['derivative_structure_sha256']})
 out.sort(key=lambda r:(-r['risk_priority_score'],r['source_key']))
 with (root/'SOURCE_VISUAL_REVIEW_PRIORITY_V1.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,out[0].keys());w.writeheader();w.writerows(out)
 q={'source_count':len(out),'requires_native_scene_review':len(out),'priority_gt_5':sum(x['risk_priority_score']>5 for x in out),
    'none_approved_for_training':True,'largest_mesh_object_count':max(x['mesh_object_count'] for x in out),
    'largest_topological_island_count':max(x['topology_island_count'] for x in out)}
 (root/'VISUAL_REVIEW_QUEUE_SUMMARY_V1.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':run(pathlib.Path(sys.argv[1]))