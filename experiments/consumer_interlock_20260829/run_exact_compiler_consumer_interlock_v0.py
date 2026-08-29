from __future__ import annotations
import base64, hashlib, json, math, zlib
from pathlib import Path
import numpy as np

from realsas_compiler_core.types import (
    SurfaceNode, RiggingSurfaceIR,
    SkeletonProposalJoint, SkeletonProposalEdge, SkeletonProposalIR,
)
from realsas_compiler_core.rig import qualify_skeleton
from realsas_compiler_core.skin import qualify_skin
from realsas_compiler_core.product import assemble_product, bind_proof, project_runtime_package

HERE=Path(__file__).resolve().parent
EXPECTED_FIXTURE_TRANSPORT_SHA='81634b7db6dbae3f7cc30d7f6942ace9be841416dd1db833185884d3e10584c5'
EXPECTED_VENDOR_RAW_SHA='3a6076b30e0a23807f952365d39d81ddf5d4b1dba734c0bdba47567bced26850'

def load_fixture():
    p=HERE/'CONSUMER_INTERLOCK_COMPILER_FIXTURE_V2.b64'
    transport=p.read_text().strip()
    if hashlib.sha256(transport.encode('ascii')).hexdigest()!=EXPECTED_FIXTURE_TRANSPORT_SHA:
        raise RuntimeError('fixture transport drift')
    raw=zlib.decompress(base64.b64decode(transport,validate=True))
    x=json.loads(raw)
    if x['schema']!='RealSaS.ConsumerInterlock.CompilerFixture.v2': raise RuntimeError('fixture schema drift')
    return x

def reconstruct(x):
    s=x['surface']
    nodes=tuple(SurfaceNode(n['surface_id'],tuple(map(float,n['P'])),tuple(map(int,n['support_views'])),(),(),metadata={'fixture':'CLEAN_D2_512'}) for n in s['nodes'])
    surf=RiggingSurfaceIR(nodes,(),s['geometry_lineage_hash'],s['builder_id'],s['schema_version'],{'fixture_asset':x['asset_id'],'d2_support_pairs':x['d2_support_pairs']})
    g=x['g0']
    joints=tuple(SkeletonProposalJoint(j['proposal_id'],tuple(map(float,j['position'])),float(j['root_score']),float(j['confidence']),tuple(j['support_surface_ids']),dict(j['metadata'])) for j in g['joints'])
    edges=tuple(SkeletonProposalEdge(e['edge_id'],e['parent_proposal_id'],e['child_proposal_id'],float(e['score']),float(e['confidence']),bool(e['hard_required']),bool(e['hard_forbidden']),e['reason'],dict(e['metadata'])) for e in g['edges'])
    prop=SkeletonProposalIR(joints,edges,g['surface_binding_hash'],g['model_provenance'],g['schema_version'],dict(g['metadata']))
    return surf,prop

def a0(surface,skeleton,topk=4,sigma_scale=.35):
    from realsas_compiler_core.types import SkinInfluenceProposal,SkinProposalIR
    SP=np.asarray([n.P for n in surface.surface_nodes],np.float32)
    JP=np.asarray([j.position for j in skeleton.joints],np.float32)
    JID=[j.canonical_joint_id for j in skeleton.joints]
    diag=float(np.linalg.norm(SP.max(0)-SP.min(0))); sigma=max(diag*sigma_scale,1e-5)
    inf=[]
    for n,p in zip(surface.surface_nodes,SP):
        d=np.linalg.norm(JP-p[None],axis=1); idx=np.argsort(d)[:min(topk,len(d))]
        q=np.exp(-.5*(d[idx]/sigma)**2); q=np.maximum(q,1e-12); w=q/q.sum()
        for ii,ww in zip(idx,w): inf.append(SkinInfluenceProposal(n.surface_id,JID[int(ii)],float(ww)))
    return SkinProposalIR(tuple(inf),surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,'SACRIFICIAL_A0_DISTANCE_SOFTMAX_V0',metadata={'topk':topk,'sigma_scale':sigma_scale,'canonical_authority':False})

def rot_axis_angle(axis,theta):
    a=np.asarray(axis,float); a=a/max(np.linalg.norm(a),1e-12); x,y,z=a; c=math.cos(theta); s=math.sin(theta); C=1-c
    return np.array([[c+x*x*C,x*y*C-z*s,x*z*C+y*s],[y*x*C+z*s,c+y*y*C,y*z*C-x*s],[z*x*C-y*s,z*y*C+x*s,c+z*z*C]],float)

def deformation_probe(surface,skeleton,skin):
    SP=np.asarray([n.P for n in surface.surface_nodes],float); sid_to_i={n.surface_id:i for i,n in enumerate(surface.surface_nodes)}
    joints={j.canonical_joint_id:j for j in skeleton.joints}; root=skeleton.root_id
    transforms={root:(np.eye(3),np.zeros(3))}
    for jid,j in sorted(joints.items()):
        if jid==root or j.parent_canonical_id is None: continue
        p=np.asarray(joints[j.parent_canonical_id].position,float); v=np.asarray(j.position,float)-p
        axis=np.cross(v,np.array([0.,0.,1.]))
        if np.linalg.norm(axis)<1e-6: axis=np.array([1.,0.,0.])
        R=rot_axis_angle(axis,math.radians(8.0)); transforms[jid]=(R,p-R@p)
    out=np.zeros_like(SP); mass=np.zeros(len(SP))
    for row in skin.rows:
        i=sid_to_i[row.surface_id]
        for jid,w in row.influences:
            R,t=transforms.get(jid,(np.eye(3),np.zeros(3))); out[i]+=float(w)*(R@SP[i]+t); mass[i]+=float(w)
    if not np.allclose(mass,1.0,atol=1e-6): raise RuntimeError('skin simplex drift in probe')
    d=np.linalg.norm(out-SP,axis=1); diag=float(np.linalg.norm(SP.max(0)-SP.min(0)))
    return {'mean_displacement':float(d.mean()),'p95_displacement':float(np.quantile(d,.95)),'max_displacement':float(d.max()),'bbox_diag':diag,'finite':bool(np.all(np.isfinite(out))),'nontrivial':bool(d.mean()>1e-5),'bounded':bool(d.max()<.35*diag)}

def main():
    x=load_fixture(); surface,g0=reconstruct(x)
    qsk=qualify_skeleton(surface,g0,run_ilp_shadow=False)
    proposal_ids={j.proposal_id for j in g0.joints}; canonical_ids={j.canonical_joint_id for j in qsk.joints}
    if len(qsk.joints)!=len(g0.joints): raise RuntimeError('required G0 joints were not all admitted')
    if proposal_ids & canonical_ids: raise RuntimeError('proposal IDs leaked into canonical IDs')
    if not all(j.startswith('J:') for j in canonical_ids): raise RuntimeError('Compiler did not mint J:* canonical IDs')
    a=a0(surface,qsk); qskin=qualify_skin(surface,qsk,a,max_influences=4)
    if len(qskin.rows)!=512: raise RuntimeError('A0 skin did not qualify 512/512 rows')
    probe=deformation_probe(surface,qsk,qskin); passed=probe['finite'] and probe['nontrivial'] and probe['bounded']
    product=assemble_product(surface,qsk,qskin,deformation_state={'consumer_interlock_probe':probe},editable_metadata={'consumer_profile':'G0_A0_V0','sacrificial':True})
    proof=bind_proof(product,probe_plan={'name':'SACRIFICIAL_ROTATE_NONROOT_8DEG_V0'},measurements=probe,passed=passed,proof_payload={'consumer_interlock':True,'not_product_quality_claim':True})
    runtime=project_runtime_package(product,proof,manifest={'consumer_interlock':True},runtime_payload_ref='NO_PAYLOAD__INTERLOCK_ONLY')
    out={
      'schema':'RealSaS.ConsumerInterlock.ExactCompilerCleanResult.v1','asset_id':x['asset_id'],
      'surface_nodes':len(surface.surface_nodes),'g0_joints':len(g0.joints),'g0_edges':len(g0.edges),
      'qualified_joints':len(qsk.joints),'root_id':qsk.root_id,
      'proposal_ids_disjoint_from_canonical':not bool(proposal_ids & canonical_ids),
      'all_canonical_ids_minted_J':all(j.startswith('J:') for j in canonical_ids),
      'optimizer_solver':qsk.qualification_report.get('solver'),'optimizer_status':qsk.qualification_report.get('status'),
      'optimizer_optimality_proven':qsk.qualification_report.get('optimality_proven'),
      'qualified_skin_rows':len(qskin.rows),'skin_corrected_rows':qskin.qualification_report.get('corrected_row_count'),
      'skin_max_simplex_residual_before':qskin.qualification_report.get('max_simplex_residual_before'),
      'deformation_probe':probe,'proof_passed':proof.passed,
      'proof_bound_to_exact_product_state':proof.product_state_hash==product.product_state_hash,
      'runtime_bound_to_exact_product_state':runtime.source_product_state_hash==product.product_state_hash,
      'product_state_hash':product.product_state_hash,'skeleton_lineage_hash':qsk.skeleton_lineage_hash,'skin_lineage_hash':qskin.skin_lineage_hash,
      'vendor_raw_sha256_expected':EXPECTED_VENDOR_RAW_SHA,'status':'PASS' if passed else 'FAIL'
    }
    op=HERE/'CONSUMER_INTERLOCK_EXACT_COMPILER_CLEAN_RESULT_V1.json'; op.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))
    if not passed: raise SystemExit(2)
if __name__=='__main__': main()
