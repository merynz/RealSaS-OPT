#!/usr/bin/env python3
"""Decode *actual* canonical-GLB skin influences for MIRA/AXIS/TESSA/IRIS.

Evidence only. Proves that canonical skinned mesh VEC4 JOINTS/WEIGHTS
accessors decode to concrete numeric bind/influence data, not that original
source, semantically named limb, deformation teacher, IK control, or training
rights are established. Sparse/morph/extension paths fail explicitly to
UNKNOWN instead of inventing anatomy or silent negative labels.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA='RealSaS.UniML3DCanonicalSkinInfluence.v1'
HEADS=('IRIS_GEOMETRY','IRIS_SEMANTIC','IRIS_GRAPHICS','TESSA','AXIS','MIRA')
DTYPES={5120:('b',1,127.0),5121:('B',1,255.0),5122:('h',2,32767.0),
        5123:('H',2,65535.0),5125:('I',4,4294967295.0),5126:('f',4,1.0)}
LANES={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
MAX_GLB=32*1024*1024
MAX_VERTEX_COUNT=3_000_000


def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()


def glb(path):
 b=Path(path).read_bytes()
 if len(b)<20 or len(b)>MAX_GLB:raise ValueError('GLB_SIZE')
 magic,ver,declared=struct.unpack_from('<4sII',b,0)
 if magic!=b'glTF' or ver!=2 or declared!=len(b):raise ValueError('GLB_HEADER')
 pos=12;chunks=[]
 while pos+8<=len(b):
  n,typ=struct.unpack_from('<I4s',b,pos);pos+=8
  if n>len(b)-pos or n%4:raise ValueError('GLB_CHUNK')
  chunks.append((typ,b[pos:pos+n]));pos+=n
 if pos!=len(b) or not chunks or chunks[0][0]!=b'JSON':raise ValueError('GLB_JSON')
 data=json.loads(chunks[0][1].decode().strip(' \t\r\n\0'))
 binblock=next((x for typ,x in chunks if typ==b'BIN\0'),b'')
 return data,binblock


def decode(data,binblock,acc_id):
 accessors=data.get('accessors',[])
 if not isinstance(acc_id,int) or acc_id<0 or acc_id>=len(accessors):raise ValueError('BAD_ACCESSOR_ID')
 a=accessors[acc_id]
 if 'sparse' in a:raise ValueError('UNSUPPORTED_SPARSE_ACCESSOR')
 if 'bufferView' not in a:raise ValueError('MISSING_BUFFER_VIEW')
 views=data.get('bufferViews',[])
 if not isinstance(a['bufferView'],int) or a['bufferView']>=len(views):raise ValueError('BAD_BUFFER_VIEW')
 v=views[a['bufferView']]
 if v.get('buffer',0)!=0:raise ValueError('EXTERNAL_BUFFER_NOT_COURTED')
 comp=a.get('componentType');typ=a.get('type');count=a.get('count')
 if comp not in DTYPES or typ not in LANES or not isinstance(count,int) or count<=0 or count>MAX_VERTEX_COUNT:
  raise ValueError('UNSUPPORTED_ACCESSOR_LAYOUT')
 fmt,size,normalized_denom=DTYPES[comp];lanes=LANES[typ];packed=lanes*size
 stride=v.get('byteStride',packed)
 if not isinstance(stride,int) or stride<packed or stride>4096 or stride%size:raise ValueError('INVALID_STRIDE')
 view_off=v.get('byteOffset',0);view_size=v.get('byteLength')
 acc_off=a.get('byteOffset',0)
 if not all(isinstance(x,int) and x>=0 for x in (view_off,view_size,acc_off)):
  raise ValueError('INVALID_BUFFER_VIEW_BOUNDARY')
 off=view_off+acc_off
 if off<view_off or off+stride*(count-1)+packed>view_off+view_size or view_off+view_size>len(binblock):
  raise ValueError('BUFFER_VIEW_OOB')
 if off%size:raise ValueError('MISALIGNED_ACCESSOR')
 st=struct.Struct('<'+fmt*lanes)
 if a.get('normalized') and comp!=5126:
  return [tuple(max(-1.,x/normalized_denom) if comp in (5120,5122) else x/normalized_denom
                for x in st.unpack_from(binblock,off+i*stride)) for i in range(count)]
 return [st.unpack_from(binblock,off+i*stride) for i in range(count)]


def inspect(path):
 path=Path(path);data,binary=glb(path)
 output={'canonical_glb_sha256':file_sha(path), 'mesh_count':len(data.get('meshes',[])),
         'skin_count':len(data.get('skins',[])), 'node_count':len(data.get('nodes',[])),
         'canonical_joint_count_declared':sum(len(s.get('joints',[])) for s in data.get('skins',[])),
         'skinned_primitive_count':0,'skinned_vertex_count':0,
         'decoded_joint_weight_pairs':0,'valid_weight_vertex_count':0,
         'zero_weight_vertex_count':0,'weight_sum_not_near_one_count':0,
         'nonfinite_or_negative_weight_count':0,'out_of_skin_joint_palette_count':0,
         'positive_vertex_joint_assignments':0,'joint_influence_area_sum':0.0,
         'canonical_inverse_bind_count_matches':0,'canonical_inverse_bind_count_mismatch':0,
         'unsupported_processed_primitive_count':0,'unsupported_reasons':[],
         'actual_numeric_skin_values_observed':False,
         'source_original_bytes_parity':'UNKNOWN','source_rights':'UNKNOWN',
         'pose_deformation_teacher':'NOT_RUN','semantic_joint_names_review':'UNKNOWN',
         'mira_training_admitted':False,'axis_training_admitted':False}
 influences=defaultdict(lambda:{'count':0,'sum':0.0,'max':0.0,'source_node_name_hint':''})
 nodes=data.get('nodes',[])
 meshes=data.get('meshes',[])
 skins=data.get('skins',[])
 for skin_idx,skin in enumerate(skins):
  joints=skin.get('joints',[])
  ibm_id=skin.get('inverseBindMatrices')
  if ibm_id is None:
   output['canonical_inverse_bind_count_mismatch']+=1 # glTF implicit identity IBMs, not authored bind matrices.
  else:
   try:
    a=data['accessors'][ibm_id]
    if a.get('count')!=len(joints) or a.get('type')!='MAT4':
     output['canonical_inverse_bind_count_mismatch']+=1
    else:
     values=decode(data,binary,ibm_id)
     if all(all(math.isfinite(float(x)) for x in row) for row in values):
      output['canonical_inverse_bind_count_matches']+=1
     else:output['canonical_inverse_bind_count_mismatch']+=1
   except (ValueError,KeyError,IndexError,TypeError) as ex:
    output['canonical_inverse_bind_count_mismatch']+=1
    output['unsupported_reasons'].append('BIND:'+type(ex).__name__+':'+str(ex))
  for local_id,node_idx in enumerate(joints):
   hint=nodes[node_idx].get('name','') if isinstance(node_idx,int) and 0<=node_idx<len(nodes) else ''
   influences[(skin_idx,local_id)]['source_node_name_hint']=hint[:100]
 for node_idx,node in enumerate(nodes):
  mesh_idx=node.get('mesh');skin_idx=node.get('skin')
  if not isinstance(mesh_idx,int) or not 0<=mesh_idx<len(meshes) or not isinstance(skin_idx,int) or not 0<=skin_idx<len(skins):
   continue
  skin=skins[skin_idx];joint_count=len(skin.get('joints',[]))
  for primitive_idx,p in enumerate(meshes[mesh_idx].get('primitives',[])):
   attrs=p.get('attributes',{})
   if 'JOINTS_0' not in attrs or 'WEIGHTS_0' not in attrs:
    output['unsupported_processed_primitive_count']+=1
    output['unsupported_reasons'].append(f'NODE{node_idx}_PRIM{primitive_idx}:NO_JOINT_WEIGHT_PAIR')
    continue
   try:
    pairs=[]
    for suffix in ('0','1'):
     jkey='JOINTS_'+suffix;wkey='WEIGHTS_'+suffix
     if (jkey in attrs)!=(wkey in attrs):raise ValueError('UNPAIRED_JOINT_WEIGHT_ATTRIBUTES')
     if jkey not in attrs:continue
     j=decode(data,binary,attrs[jkey]);w=decode(data,binary,attrs[wkey])
     if len(j)!=len(w) or any(len(v)!=4 for v in j+w):raise ValueError('SKIN_ATTRIBUTE_SHAPE_MISMATCH')
     if data['accessors'][attrs[jkey]]['type']!='VEC4' or data['accessors'][attrs[wkey]]['type']!='VEC4':
      raise ValueError('JOINT_WEIGHT_MUST_BE_VEC4')
     if data['accessors'][attrs[jkey]]['componentType'] not in (5121,5123) or data['accessors'][attrs[wkey]]['componentType'] not in (5121,5123,5126):
      raise ValueError('INVALID_JOINT_WEIGHT_COMPONENT_TYPE')
     pairs.append((j,w))
    count=len(pairs[0][0]);
    if 'POSITION' in attrs and data['accessors'][attrs['POSITION']]['count']!=count:raise ValueError('SKIN_POSITION_COUNT_MISMATCH')
    for i in range(count):
     values=[(int(j[i][k]),float(w[i][k])) for j,w in pairs for k in range(4)]
     total=sum(weight for _,weight in values if math.isfinite(weight))
     bad=sum(int(not math.isfinite(weight) or weight<0) for _,weight in values)
     invalid=sum(int(weight>1e-8 and (joint<0 or joint>=joint_count)) for joint,weight in values)
     output['nonfinite_or_negative_weight_count']+=bad
     output['out_of_skin_joint_palette_count']+=invalid
     output['zero_weight_vertex_count']+=int(total<=1e-8)
     output['weight_sum_not_near_one_count']+=int(abs(total-1.)>0.02)
     output['valid_weight_vertex_count']+=int(not bad and not invalid and abs(total-1.)<=0.02)
     for joint,weight in values:
      if math.isfinite(weight) and weight>1e-8 and 0<=joint<joint_count:
       entry=influences[(skin_idx,joint)]
       entry['count']+=1;entry['sum']+=weight;entry['max']=max(entry['max'],weight)
       output['positive_vertex_joint_assignments']+=1
    output['skinned_primitive_count']+=1
    output['skinned_vertex_count']+=count
    output['decoded_joint_weight_pairs']+=len(pairs)
   except (ValueError,IndexError,KeyError,TypeError,OverflowError,struct.error) as ex:
    output['unsupported_processed_primitive_count']+=1
    output['unsupported_reasons'].append(f'NODE{node_idx}_PRIM{primitive_idx}:'+type(ex).__name__+':'+str(ex))
 output['actual_numeric_skin_values_observed']=(output['skinned_vertex_count']>0)
 output['joint_influence_area_sum']=round(sum(e['sum'] for e in influences.values()),6)
 return output,[{'skin_index':skin,'joint_palette_index':joint,'canonical_node_name_unreviewed':entry['source_node_name_hint'],
                 'positive_influence_vertex_count':entry['count'],
                 'sum_skin_weights':round(entry['sum'],6),'max_skin_weight':round(entry['max'],6),
                 'semantic_joint_class_approved':False,'source_original_bind_pose_approved':False}
                for (skin,joint),entry in sorted(influences.items())]


def run(glb_dir,manifest,out,expect_manifest=96,expect_downloaded=84):
 glb_dir,manifest,out=Path(glb_dir),Path(manifest),Path(out)
 with manifest.open('r',newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
 if len(rows)!=expect_manifest:raise ValueError('MANIFEST_COUNT_MISMATCH')
 out.mkdir(parents=True,exist_ok=True)
 records=[];per_joint=[]
 for r in rows:
  if r['download_status']!='DOWNLOADED':continue
  uid=r['object_type']
  if not uid.replace('_','').isalnum() or not uid or len(uid)>150:raise ValueError('INVALID_SOURCE_ID')
  file=glb_dir/(uid+'.glb')
  if not file.is_file() or file_sha(file)!=r['download_sha256']:raise ValueError('SOURCE_SHA_MISMATCH:'+uid)
  record,joints=inspect(file)
  record.update(provider='UNIML3D',object_type=uid,morphology_provider_claim=r.get('category_provider_claim','UNKNOWN'))
  records.append(record)
  per_joint.extend([{'source_key':'UNIML3D:'+uid,'canonical_glb_sha256':record['canonical_glb_sha256'],**x} for x in joints])
 if len(records)!=expect_downloaded:raise ValueError('DOWNLOADED_COUNT_MISMATCH')
 def write_csv(name,items):
  f=out/name
  with f.open('w',encoding='utf-8',newline='') as fp:
   writer=csv.DictWriter(fp,fieldnames=list(items[0]))
   writer.writeheader();writer.writerows(items)
  return file_sha(f)
 shas={}
 shas['UNIML3D_ACTUAL_CANONICAL_SKIN_WEIGHT_V1.csv']=write_csv('UNIML3D_ACTUAL_CANONICAL_SKIN_WEIGHT_V1.csv',records)
 shas['UNIML3D_JOINT_INFLUENCE_MATRIX_V1.csv']=write_csv('UNIML3D_JOINT_INFLUENCE_MATRIX_V1.csv',per_joint)
 summary={'schema':SCHEMA,'canonical_glb_sha_verified':len(records),
  'actual_numeric_skin_data_observed_sources':sum(bool(x['actual_numeric_skin_values_observed']) for x in records),
  'joint_palette_weight_matrix_rows':len(per_joint),
  'decoded_skinned_vertices':sum(x['skinned_vertex_count'] for x in records),
  'zero_weight_vertices':sum(x['zero_weight_vertex_count'] for x in records),
  'weight_sum_out_of_tolerance_vertices':sum(x['weight_sum_not_near_one_count'] for x in records),
  'invalid_palette_weight_influences':sum(x['out_of_skin_joint_palette_count'] for x in records),
  'unsupported_primitive_count':sum(x['unsupported_processed_primitive_count'] for x in records),
  'inverse_bind_count_matches':sum(x['canonical_inverse_bind_count_matches'] for x in records),
  'source_original_bytes_courted':0,'commercial_ml_training_rights_courted':0,
  'mira_deformation_teacher_pass':0,'axis_controller_proof_pass':0,'iris_graphics_2d_art_pass':0,
  'per_head_training_admission':{h:0 for h in HEADS},'training_authorized':False,
  'input_manifest_sha256':file_sha(manifest),'output_sha256':shas}
 (out/'SUMMARY_V1.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 return summary


def main():
 a=argparse.ArgumentParser()
 a.add_argument('--glb-dir',type=Path,required=True)
 a.add_argument('--manifest',type=Path,required=True)
 a.add_argument('--out',type=Path,required=True)
 opts=a.parse_args();print(json.dumps(run(opts.glb_dir,opts.manifest,opts.out),indent=2))
if __name__=='__main__':main()