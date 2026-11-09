#!/usr/bin/env python3
"""Per-source array court of historical normalized.npz: derived mechanical evidence only.

This never observes the creator's original meshes, textures, rig animation or license.
No UDF/SDF sign, source-to-creator parity, TESSA teacher, or training authority minted.
"""
import argparse,csv,gzip,hashlib,json,math,pathlib,collections,zipfile
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REQUIRED={"vertices_source","vertices","faces","vertex_normals","canonical_transform","inverse_canonical_transform","parents","deform_mask","skin","rest_local_source","rest_world_source","bone_heads","bone_tails"}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sf(x):return float(x) if math.isfinite(float(x)) else None
def topology(v,f):
  """Full vertex-index edge audit: no 3D sign interpretation."""
  if len(f)==0:return dict(boundary_edges=0,nonmanifold_edges=0,winding_disagreement_edges=0,unique_edges=0)
  edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]],axis=0)
  sorted_edges=np.sort(edges,axis=1)
  uniq,inv,counts=np.unique(sorted_edges,axis=0,return_inverse=True,return_counts=True)
  # For exactly two adjacent triangles, directions should be opposite.
  directed=np.asarray(edges[:,0]<edges[:,1],dtype=np.int64)
  both=np.bincount(inv,weights=directed,minlength=len(uniq))
  return {'unique_edges':len(uniq),'boundary_edges':int(np.count_nonzero(counts==1)),
          'nonmanifold_edges':int(np.count_nonzero(counts>2)),
          'winding_disagreement_edges':int(np.count_nonzero((counts==2)&(both!=1)))}
def court(path,structure=None):
  if path.stat().st_size>400*1024*1024:raise ValueError('ARCHIVE_COMPRESSED_TOO_LARGE')
  with zipfile.ZipFile(path) as zz:
    members=zz.infolist()
    if len(members)>70 or sum(x.file_size for x in members)>900*1024*1024:
      raise ValueError('UNBOUNDED_NPZ_RESOURCE')
  result={'normalized_npz_sha256':sha(path),'authority':'HISTORICAL_NORMALIZED_DERIVATIVE_NOT_ORIGINAL_MESH',
          'teacher_training_admission':False,'global_sdf_sign_admission':False,'2d_art_approval':False}
  with np.load(path,allow_pickle=False) as z:
    keys=set(z.files);missing=REQUIRED-keys
    result['array_names']=sorted(keys)
    if missing:raise ValueError('MISSING_REQUIRED_ARRAYS:'+','.join(sorted(missing)))
    v=np.asarray(z['vertices']);orig=np.asarray(z['vertices_source']);f=np.asarray(z['faces']);norm=np.asarray(z['vertex_normals']);
    P=np.asarray(z['parents']);W=np.asarray(z['skin']);deform=np.asarray(z['deform_mask']);
    C=np.asarray(z['canonical_transform']);CI=np.asarray(z['inverse_canonical_transform']);
    rl=np.asarray(z['rest_local_source']);rw=np.asarray(z['rest_world_source']);heads=np.asarray(z['bone_heads']);tails=np.asarray(z['bone_tails']);
    n=len(v);nf=len(f);nj=len(P)
    if not 1<=n<=700000 or not 1<=nf<=1100000 or not 1<=nj<=1024:raise ValueError('ARRAY_DOMAIN_TOO_LARGE_OR_EMPTY')
    shapes={'vertices':(n,3),'vertices_source':(n,3),'faces':(nf,3),'vertex_normals':(n,3),
            'skin':(n,nj),'parents':(nj,),'deform_mask':(nj,),
            'canonical_transform':(4,4),'inverse_canonical_transform':(4,4),
            'rest_local_source':(nj,4,4),'rest_world_source':(nj,4,4),'bone_heads':(nj,3),'bone_tails':(nj,3)}
    for k,shape in shapes.items():
      arr=z[k]
      if arr.shape!=shape or arr.dtype.kind not in 'buif':raise ValueError('ARRAY_TYPE_OR_SHAPE:'+k)
    for k in ['vertices','vertices_source','vertex_normals','skin','canonical_transform','inverse_canonical_transform','rest_local_source','rest_world_source','bone_heads','bone_tails']:
      if not np.isfinite(z[k]).all():raise ValueError('NONFINITE_ARRAY:'+k)
    if not np.issubdtype(f.dtype,np.integer) or not np.issubdtype(P.dtype,np.integer):raise ValueError('INDICES_NOT_INTEGERS')
    if np.any(f<0) or np.any(f>=n):raise ValueError('FACE_INDEX_OUT_OF_BOUNDS')
    if np.any(P < -1) or np.any(P>=nj):raise ValueError('PARENT_OUT_OF_BOUNDS')
    if np.any(P==np.arange(nj)):raise ValueError('JOINT_SELF_PARENT')
    if not np.any(P==-1):raise ValueError('NO_ROOT_JOINT')
    # Universal finite cycle detection independent of bone name semantics.
    for j in range(nj):
      curr=j;seen=set()
      while curr!=-1:
        if curr in seen:raise ValueError('PARENT_CYCLE')
        seen.add(curr);curr=int(P[curr])
    coord=np.concatenate([orig.astype(np.float64),np.ones((n,1),dtype=np.float64)],axis=1)
    transformed=coord@C.T
    e=float(np.max(np.abs(transformed[:,:3]-v)))
    ident=float(np.max(np.abs(np.asarray(CI,dtype=np.float64)@np.asarray(C,dtype=np.float64)-np.eye(4))))
    normal_lengths=np.linalg.norm(norm,axis=1)
    rs=W.astype(np.float64).sum(axis=1)
    zero=rs<=1e-8
    bad=(rs>1e-8)&(np.abs(rs-1)>1e-3)
    neg=int(np.count_nonzero(W < -1e-8))
    influence=np.count_nonzero(W>1e-6,axis=1)
    counts=np.count_nonzero(W>1e-6,axis=0)
    mass=W.sum(axis=0,dtype=np.float64)
    # Triangle degeneracy from actual source-normalized 3D positions.
    a=v[f[:,1]]-v[f[:,0]];b=v[f[:,2]]-v[f[:,0]]
    double_area=np.linalg.norm(np.cross(a,b),axis=1)
    diag=float(np.linalg.norm(v.max(axis=0)-v.min(axis=0)))
    deg=int(np.count_nonzero(double_area<=max(diag*diag*1e-12,1e-18)))
    # O(N log N) exact edge-topology audit; no volume sign extrapolation.
    topo=topology(v,f)
    # Vertex-index graph components; no union of coincident but distinct indices.
    face_edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]],axis=0)
    sparse_graph=coo_matrix((np.ones(len(face_edges),dtype=np.int32),(face_edges[:,0],face_edges[:,1])),shape=(n,n)).tocsr()
    component_count,component_labels=connected_components(sparse_graph,directed=False)
    face_component_count=int(np.unique(component_labels[f[:,0]]).size)
    used=np.zeros(n,dtype=bool);used[f.ravel()]=True
    isolated=int(np.count_nonzero(~used))
    lengths=np.linalg.norm(tails-heads,axis=1)
    result.update({'vertex_count':n,'triangle_count':nf,'joint_count':nj,
      'source_space_vertex_count':len(orig),'canonical_roundtrip_max_abs_err':ident,
      'source_to_canonical_max_abs_err':e,
      'vertex_normal_nonunit_count':int(np.count_nonzero((normal_lengths>1e-6)&(np.abs(normal_lengths-1)>0.005))),
      'zero_normal_count':int(np.count_nonzero(normal_lengths<=1e-6)),
      'normal_norm_p99_abs_error':float(np.percentile(np.abs(normal_lengths-1),99)),
      'degenerate_triangle_count':deg,'canonical_bbox_diagonal':diag,
      'vertex_index_connected_component_count':int(component_count),
      'face_supported_component_count':face_component_count,
      'isolated_vertex_count':isolated,
      'indexed_euler_characteristic':int(n-topo['unique_edges']+nf),
      **topo,
      'joint_root_count':int(np.count_nonzero(P==-1)),'deforming_joint_count':int(np.count_nonzero(deform)),
      'skin_zero_row_count':int(np.count_nonzero(zero)),'skin_invalid_sum_count':int(np.count_nonzero(bad)),
      'skin_negative_entries':neg,'skin_max_influences':int(np.max(influence)),
      'skin_mean_influences':float(np.mean(influence)),'skin_p99_influences':float(np.percentile(influence,99)),
      'skin_unreferenced_joint_count':int(np.count_nonzero(counts==0)),
      'bone_zero_length_count':int(np.count_nonzero(lengths<=1e-9)),
      'structure_vertex_parity':'UNKNOWN_NO_STRUCTURE','structure_face_parity':'UNKNOWN_NO_STRUCTURE',
      'structure_bone_count_parity':'UNKNOWN_NO_STRUCTURE'})
    if structure:
      ms=structure.get('mesh_objects') or []
      if isinstance(ms,list) and all(isinstance(x,dict) for x in ms):
        try:
          totalv=sum(int(x['vertex_count']) for x in ms);totalf=sum(int(x['face_count']) for x in ms)
          result['structure_vertex_parity']=bool(totalv==n)
          result['structure_face_parity']=bool(totalf==nf)
          bc=structure.get('bone_count')
          if bc is not None:result['structure_bone_count_parity']=bool(int(bc)==nj)
        except(TypeError,ValueError,KeyError):result['structure_vertex_parity']='INVALID_STRUCTURE_COUNTS'
    joint=[{'joint_index':i,'parent_index':int(P[i]),'deform_mask':int(deform[i]),
            'influenced_vertices_count':int(counts[i]),'summed_skin_weight_mass':float(mass[i]),
            'head_tail_length':float(lengths[i])} for i in range(nj)]
    return result,joint

def main():
 p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--baseline-inventory',required=True);p.add_argument('--variant-dir',required=True);p.add_argument('--out',required=True)
 a=p.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 with open(a.queue,newline='') as f:queue=list(csv.DictReader(f))
 statuses={}
 with gzip.open(a.baseline_inventory,'rt',newline='') as f:
  for row in csv.DictReader(f):
   if row['pool']=='MASTER3993':statuses[row['source_key']]=row['master_v11_source_recovery_status']
 results=[];joints=[];status_count=collections.Counter()
 for row in queue:
  key=row['source_key'];status=statuses[key]
  if status=='PROTECTED_HOLDOUT_ORIGINAL_NOT_TOUCHED':continue
  # Only the historical recovered original-source SHA-verified cohort, not merely path matches.
  if status!='SHA_VERIFIED_RECOVERY_RECEIPT':continue
  folder=pathlib.Path(a.variant_dir)/row['candidate_id'].replace(':','_',1)
  npz=folder/'normalized.npz'
  if not npz.exists():
   res={'court':'MISSING_HISTORICAL_NORMALIZED_DERIVATIVE','source_key':key};status_count[res['court']]+=1;results.append(res);continue
  try:
   st=json.loads((folder/'source_structure.json').read_text()) if (folder/'source_structure.json').exists() else None
   v,j=court(npz,st)
   v['court']='NUMERIC_DERIVATIVE_OBSERVATION_ONLY';v['source_key']=key;v['candidate_id']=row['candidate_id'];v['source_structure_sha256']=sha(folder/'source_structure.json') if st else None
   status_count[v['court']]+=1
   for x in j:x['source_key']=key
   joints.extend(j);results.append(v)
  except Exception as e:
   res={'source_key':key,'court':'QUARANTINED_BAD_HISTORICAL_NPZ','error_kind':type(e).__name__,'error_detail':str(e)[:180]}
   status_count[res['court']]+=1;results.append(res)
 with gzip.open(out/'MASTER726_NORMALIZED_DERIVATIVE_PER_ASSET_V13.jsonl.gz','wt') as f:
  for x in results:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
 with gzip.open(out/'MASTER726_NORMALIZED_DERIVATIVE_JOINTS_V13.csv.gz','wt',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['source_key','joint_index','parent_index','deform_mask','influenced_vertices_count','summed_skin_weight_mass','head_tail_length']);w.writeheader();w.writerows(joints)
 summary={'schema':'RealSaS.HistoricalNormalizedDerivativeV13','candidate_attempts':len(results),'results':dict(status_count),
          'per_joint_rows':len(joints),'original_source_byte_rehash_admission':0,'global_sdf_sign_admission':0,
          'mechanically_qualified_tessa_teacher_admission':0,'rights_admission':0,'training_authorized':0,
          'protected_holdout_folders_visited':0}
 (out/'SUMMARY_NORMALIZED_V13.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()