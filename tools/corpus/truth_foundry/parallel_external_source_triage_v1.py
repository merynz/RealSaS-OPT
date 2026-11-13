#!/usr/bin/env python3
"""RealSaS: independent external-pool candidate queues, NOT source truth.

Consumes published provider metadata only; does not download 3D bytes or grant
source license, morphology, renderer, or any model-head admission.
"""
import argparse, csv, hashlib, json, re
from collections import Counter
from pathlib import Path
import pandas as pd

SCHEMA='RealSaS.ExternalParallelSourceTriage.v1'
CATEGORIES=['rare_morphology','creature_animal','humanoid','mechanical_vehicle','unknown']
# Conservation rule: all eligible minority morphologies enter the 2,500 candidate
# queue before common bipeds, while **never** bypassing source/license/visual courts.
# An uncertain label is NOT proof of non-bipedal anatomy.
RARE_NON_BIPEDAL=('serpentine','marine','insectoid','avian','quadrupedal','articulated_rigid')
MORPH_PRIORITY={name:index for index,name in enumerate(RARE_NON_BIPEDAL)}
MORPH_PRIORITY.update({'bipedal':len(RARE_NON_BIPEDAL),
                       'uncertain':len(RARE_NON_BIPEDAL)+1})

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def decide_articulation(row):
 c=str(row.get('category_label') or '').lower()
 if any(s in c for s in ['scanned data','anatomy','furniture','clothing','architecture','interior']):return 'exclude_or_manual_review'
 if any(s in c for s in ['animal','mythical creature']):return 'creature_animal'
 if any(s in c for s in ['character','anthropomorphic']):return 'humanoid'
 if any(s in c for s in ['vehicle','robot','mechanical']):return 'mechanical_vehicle'
 return 'unknown'

def build_articulation(path,master_sha):
 a=pd.read_csv(path,keep_default_na=False)
 assert len(a)==48637 and not a.uuid.duplicated().any()
 for col in ['vertex_count','face_count','bone_count','joint_count']:
  a[col]=pd.to_numeric(a[col],errors='coerce').fillna(0).astype(int)
 a['candidate_family']=a.apply(decide_articulation,axis=1)
 a['candidate_usable_numeric']=((a.vertex_count>=300)&(a.face_count>=250)&(a.bone_count>=2)&(a.joint_count>=2)&(a.vertex_count<=200000)&(a.face_count<=400000))
 a['source_sha256_provider_claim']=''
 a['exact_master_sha_crosswalk']='NOT_AVAILABLE_FROM_PROVIDER_METADATA'
 a['scene_file_url']=a.fileIdentifier
 a['train_only_source_proof']='UNKNOWN__DATASET_TRAIN_TEST_NOT_CROSSED'
 a['morphology_visual_court']='UNKNOWN'
 a['source_rights_court']='UNKNOWN'
 a['topology_native_court']='UNKNOWN'
 a['source_identity_court']='UNKNOWN'
 a['training_authorized']=False
 # Strong rare creature and nonhumanoid priority; reserve comparable other groups.
 quotas={'creature_animal':1600,'humanoid':700,'mechanical_vehicle':150,'unknown':50}
 q=[]
 for family,limit in quotas.items():
  eligible=a[(a.candidate_family==family)&a.candidate_usable_numeric].copy()
  eligible['size_penalty']=(eligible.vertex_count/25000-1).abs().clip(lower=0)
  eligible=eligible.sort_values(['size_penalty','vertex_count','uuid'],ascending=[True,True,True])
  q.append(eligible.head(limit))
 out=pd.concat(q,ignore_index=True)
 # If no vehicle evidence in provider, remainder goes to more qualifying creatures/humanoids, not fabricated vehicles.
 if len(out)<2500:
  remaining=a[(a.candidate_usable_numeric)&(a.candidate_family.isin(['creature_animal','humanoid','mechanical_vehicle','unknown']))&(~a.uuid.isin(out.uuid))].copy()
  remaining['fam_rank']=remaining.candidate_family.map({'creature_animal':0,'mechanical_vehicle':1,'humanoid':2,'unknown':3})
  remaining['size_penalty']=(remaining.vertex_count/25000-1).abs()
  remaining=remaining.sort_values(['fam_rank','size_penalty','uuid'])
  out=pd.concat([out,remaining.head(2500-len(out))],ignore_index=True)
 assert len(out)==2500 and out.uuid.nunique()==2500
 return a,out

def build_uniml(path):
 d=pd.read_csv(path,keep_default_na=False)
 assert len(d)==7430 and not d[['dataset','object_type']].duplicated().any()
 d['candidate_morphology']=d.category
 d['source_rights_court']='UNKNOWN'
 d['scene_face_topology_court']='UNKNOWN'
 d['rest_canonical_parity']='UNKNOWN'
 d['original_asset_sha256']='UNKNOWN'
 d['training_authorized']=False
 d['source_training_subset']='PROVIDER_OBJAVERSE_ONLY' # Avoid Truebones/Mixamo unknown rights in selection
 d['candidate_usable_motion']=((pd.to_numeric(d.num_joints,errors='coerce')>=3)&(pd.to_numeric(d.num_clips,errors='coerce')>=1))
 # Add all non-bipedal Objaverse, followed by bipedal to 2500. Diversity is NOT comparable to all-human random selection.
 pool=d[(d.dataset=='objaverse')&d.candidate_usable_motion].copy()
 pool['morph_rarity_priority']=pool.category.map(MORPH_PRIORITY).fillna(len(MORPH_PRIORITY)+1).astype(int)
 pool=pool.sort_values(['morph_rarity_priority','num_clips','num_joints','object_type'],ascending=[True,False,False,True])
 out=pool.head(2500).copy()
 assert len(out)==2500 and (out.dataset=='objaverse').all()
 assert not out[['dataset','object_type']].duplicated().any()
 # Every usable rare source that fits capacity must be preserved; this is
 # a candidate queue guarantee, never a supervision quality shortcut.
 rare_pool=set(pool.loc[pool.category.isin(RARE_NON_BIPEDAL),'object_type'])
 rare_selected=set(out.loc[out.category.isin(RARE_NON_BIPEDAL),'object_type'])
 if len(rare_pool)<=2500:
  assert rare_pool<=rare_selected, 'RARE_MORPHOLOGY_DROPPED_IN_SELECTION'
 return d,out

def run(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 roster=pd.read_csv(root/'SOURCE_RECORD_CENSUS_4363.csv',keep_default_na=False)
 assert len(roster)==4363
 known={s for s in roster.source_sha256 if re.fullmatch('[0-9a-f]{64}',str(s))}
 a,aq=build_articulation(root/'meta_Articulation_XL_2.0.csv',known)
 u,uq=build_uniml(root/'skeletons.csv')
 a[['uuid','source','category_label','candidate_family','candidate_usable_numeric','vertex_count','face_count','joint_count','bone_count','fileType','fileIdentifier','source_rights_court','source_identity_court','training_authorized']].to_csv(out/'ARTICULATION_48637_WEAK_SOURCE_INVENTORY.csv',index=False)
 aq.to_csv(out/'ARTICULATION_2500_SOURCE_CANDIDATE_QUEUE.csv',index=False)
 u.to_csv(out/'UNIML3D_7430_MORPHOLOGY_SOURCE_INVENTORY.csv',index=False)
 uq.to_csv(out/'UNIML3D_2500_DIVERSITY_SOURCE_CANDIDATE_QUEUE.csv',index=False)
 art_group=aq.candidate_family.value_counts().to_dict();uniml_group=uq.category.value_counts().to_dict()
 summary={
 'schema':SCHEMA,
 'master_existing_records':4363,
 'master_known_sha_rows':sum(bool(re.fullmatch('[0-9a-f]{64}',str(s))) for s in roster.source_sha256),
 'articulation_metadata_records':len(a), 'articulation_queue':len(aq),
 'articulation_candidate_groups':art_group,'articulation_provider_source_counts':a.source.value_counts().to_dict(),
 'uniml_metadata_records':len(u),'uniml_queue':len(uq),
 'uniml_selected_morphology':uniml_group,
 'uniml_all_morphology':u.category.value_counts().to_dict(),
 'uniml_selected_origins':uq.dataset.value_counts().to_dict(),
 'rare_morphology_policy':'PRESERVE_ALL_USABLE_NON_BIPEDAL_BEFORE_BIPEDAL__SUBJECT_TO_RIGHTS_SOURCE_AND_VISUAL_COURTS',
 'non_bipedal_categories':list(RARE_NON_BIPEDAL),
 'uniml_eligible_non_bipedal_candidates':int(((u.dataset=='objaverse') & u.candidate_usable_motion & u.category.isin(RARE_NON_BIPEDAL)).sum()),
 'uniml_selected_non_bipedal_candidates':int(uq.category.isin(RARE_NON_BIPEDAL).sum()),
 'uniml_selected_bipedal_candidates':int((uq.category=='bipedal').sum()),
 'uniml_selected_uncertain_candidates':int((uq.category=='uncertain').sum()),
 'articulation_original_source_sha_crosswalk_verified':0,
 'uniml_original_source_sha_crosswalk_verified':0,
 'new_raw_asset_bytes_downloaded':0,
 'head_truth_admitted':0,
 'independent_rights_pass':0,
 'training_authorized':False,
 'candidate_only':True,
 'input_sha256':{n:digest(root/n) for n in ['meta_Articulation_XL_2.0.csv','skeletons.csv','SOURCE_RECORD_CENSUS_4363.csv']},
 'output_sha256':{p.name:digest(p) for p in out.glob('*.csv')},
 }
 (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 print(json.dumps(run(a.root,a.out),indent=2,ensure_ascii=False))