from __future__ import annotations

import argparse, copy, json, shutil
from dataclasses import replace
from pathlib import Path

from compiler.realsas_compiler_core.appearance_authority_v2 import complete_appearance_asset_from_dict
from compiler.realsas_compiler_core.artifact_codec_v2 import canonical_mesh_candidate_from_dict, read_json
from compiler.realsas_compiler_core.output_presentation_v1 import output_direction_set_from_dict
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import build_mechanical_carrier_evidence_v1
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR,
    build_appearance_domain,
    static_mesh_qualification_hash,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import visual_mesh_set_from_dict
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import sha256_file, write_ir
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    preregister_caa_backend_stage,
    compile_caa_stage,
    seal_caa_compile_stage,
    bake_complete_appearance_stage,
    qualify_complete_appearance_stage,
)

SEED_ID='SUBJECT2_KNIGHT_CORRECTED_MECHANICAL_CAA_HARMONIC_FIRST_IMMUTABLE_RECOVERED_V1_643714DE'
MAT_ID='KNIGHT_LATEST_CARRIER_MATERIALIZATION_20261010'


def row(ledger, sid):
    return next(x for x in ledger['stages'] if x['id']==sid)


def install_outputs(ledger, sid, outputs, status='PASS_DEMO_ONLY'):
    r=row(ledger,sid); r['status']=status; r['outputs']=list(outputs); r['blockers']=[]


def output_from_path(path:Path, schema:str, authority:str):
    return {'path':str(path.resolve()),'sha256':sha256_file(path),'authority_class':authority,'schema':schema}


def seed_stage(seed_ledger, child_ledger, sid):
    src=row(seed_ledger,sid)
    if str(src.get('status')) not in {'PASS','PASS_DEMO_ONLY'} or not src.get('outputs'):
        raise RuntimeError(f'SEED_STAGE_UNAVAILABLE::{sid}')
    install_outputs(child_ledger,sid,copy.deepcopy(src['outputs']),str(src['status']))


def run_stage(ctx,sid,fn):
    ctx['stage']={'id':sid}
    result=fn(ctx)
    status=str(result.get('status') or '')
    print('CAA_STAGE',sid,status,json.dumps(result.get('diagnostics') or {},sort_keys=True)[:8000],flush=True)
    if status not in {'PASS','PASS_DEMO_ONLY'}:
        raise RuntimeError(f'CAA_STAGE_FAIL::{sid}::{status}::{result.get("blockers")}')
    install_outputs(ctx['ledger'],sid,result.get('outputs') or (),status)
    (ctx['run_root']/ 'ACTIVE_RUN_V2.json').write_text(json.dumps(ctx['ledger'],indent=2,sort_keys=True)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--authority-root',type=Path,required=True); ap.add_argument('--run-id',required=True)
    a=ap.parse_args(); auth=a.authority_root.resolve(); seed=auth/'runs'/SEED_ID; mat=auth/'runs'/MAT_ID; child=auth/'runs'/a.run_id
    if child.exists(): shutil.rmtree(child)
    child.mkdir(parents=True)
    seed_manifest=json.loads((seed/'run_manifest.json').read_text())
    seed_ledger=json.loads((seed/'ACTIVE_RUN_V2.json').read_text())
    manifest=copy.deepcopy(seed_manifest)
    manifest['demo_execution']={'stage13_scientific_pass':False,'product_authority_claimed':False,'research_only_latest_carrier_visual_witness':True}
    (child/'run_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    ledger=copy.deepcopy(seed_ledger); ledger['run_id']=a.run_id; ledger['execution_class']='DEMO_WITNESS'; ledger['status']='ACTIVE'
    for r in ledger.get('stages',[]):
        if int(r.get('ordinal',0))>=18:
            r['status']='PENDING'; r['outputs']=[]; r['blockers']=[]
    for sid in ('05_CAMERA_CONTRACT_SOLVED','07_OBSERVATION_CONTRACT_QUALIFIED','15_RIGGING_SURFACE_QUALIFIED','16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED','17_MECHANICAL_PARTITION_QUALIFIED'):
        seed_stage(seed_ledger,ledger,sid)

    candidate_path=mat/'rebound_candidate.json'; addressing_path=mat/'surface_addressing.json'
    candidate=canonical_mesh_candidate_from_dict(read_json(candidate_path)); addressing=surface_addressing_from_dict(read_json(addressing_path))
    visual_out=next(o for o in row(seed_ledger,'18_CANONICAL_MESH_ADDRESSING_BUILD')['outputs'] if o.get('schema')=='RealSaS.VisualMeshSetIR.v1')
    visual_path=Path(visual_out['path']); visual=visual_mesh_set_from_dict(read_json(visual_path))
    directions_out=next(o for o in row(ledger,'16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED')['outputs'] if o.get('schema')=='RealSaS.OutputPresentationDirectionSetIR.v1')
    directions=output_direction_set_from_dict(read_json(directions_out['path']))
    domain=build_appearance_domain(candidate,addressing,directions.direction_set_hash,visual_mesh_set_hash=visual.set_hash,visual_mesh_face_count=sum(int(v.face_count) for v in visual.views))
    stage18=child/'artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD'; stage18.mkdir(parents=True)
    domain_out=write_ir(stage18/'appearance_domain.json',domain,authority_class='RESEARCH_LATEST_CARRIER_SOURCE_VISUAL_APPEARANCE_DOMAIN')
    policy_out=next(o for o in row(seed_ledger,'18_CANONICAL_MESH_ADDRESSING_BUILD')['outputs'] if o.get('schema')=='RealSaS.MeshQualificationPolicyIR.v1')
    rel=[o for o in row(seed_ledger,'18_CANONICAL_MESH_ADDRESSING_BUILD')['outputs'] if o.get('schema')=='RealSaS.RelationParentQualityReport.v1']
    outs=[output_from_path(candidate_path,'RealSaS.CanonicalMeshCandidateIR.v1','RESEARCH_LATEST_CARRIER_REBOUND'),copy.deepcopy(policy_out)]
    outs += copy.deepcopy(rel)
    outs += [output_from_path(addressing_path,'RealSaS.SurfaceAddressingIR.v1','RESEARCH_LATEST_CARRIER_ADDRESSING'),copy.deepcopy(visual_out),domain_out]
    install_outputs(ledger,'18_CANONICAL_MESH_ADDRESSING_BUILD',outs,'PASS_DEMO_ONLY')

    static=StaticCanonicalMeshQualificationIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash=addressing.addressing_hash,
        appearance_domain_binding_hash=domain.domain_hash,
        geometry_gate_binding_hash=content_sha256({'research_latest_carrier':candidate.candidate_lineage_hash}),
        partition_binding_hash=str(read_json(mat/'mechanical_partition.json')['partition_lineage_hash']),
        qualification_report={
            'status':'DEMO_ONLY_MEASURED_STATIC_CANONICAL_CARRIER__P999_FAIL',
            'demo_geometry_lineage':True,
            'product_authority_claimed':False,
            'demo_only_source_fidelity_admission':True,
            'research_latest_carrier_materialization':True,
            'xyz_topology_exact_preserved':True,
            'full_stage19_product_requalification_performed':False,
        },qualification_hash='',metadata={'research_only':True,'product_authority_claimed':False})
    static=replace(static,qualification_hash=static_mesh_qualification_hash(static))
    carrier=build_mechanical_carrier_evidence_v1(candidate,static_qualification=static,surface_addressing=addressing)
    stage19=child/'artifacts/19_STATIC_CANONICAL_MESH_QUALIFIED'; stage19.mkdir(parents=True)
    s_out=write_ir(stage19/'static_canonical_mesh_qualification.json',static,authority_class='DEMO_WITNESS_STATIC_CARRIER_BINDING')
    c_out=write_ir(stage19/'mechanical_carrier_evidence.json',carrier,authority_class='RESEARCH_LATEST_CARRIER_EVIDENCE')
    install_outputs(ledger,'19_STATIC_CANONICAL_MESH_QUALIFIED',[s_out,c_out],'PASS_DEMO_ONLY')
    (child/'ACTIVE_RUN_V2.json').write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')

    ctx={'repo_root':Path('.').resolve(),'authority_root':auth,'run_root':child,'run_id':a.run_id,'run_manifest_path':child/'run_manifest.json','run_manifest':manifest,'ledger':ledger,'stage':{'id':'INIT'}}
    for sid,fn in (
        ('20_CAA_BACKEND_PREREGISTERED',preregister_caa_backend_stage),
        ('21_CAA_COMPILE',compile_caa_stage),
        ('22_CAA_COMPILE_SEALED',seal_caa_compile_stage),
        ('23_COMPLETE_APPEARANCE_ASSET_BAKED',bake_complete_appearance_stage),
        ('24_COMPLETE_APPEARANCE_QUALIFIED',qualify_complete_appearance_stage),
    ): run_stage(ctx,sid,fn)
    asset_out=next(o for o in row(ledger,'23_COMPLETE_APPEARANCE_ASSET_BAKED')['outputs'] if o.get('schema')=='RealSaS.CompleteAppearanceAssetIR.v2')
    asset=complete_appearance_asset_from_dict(read_json(asset_out['path']))
    receipt={'schema':'RealSaS.KnightLatestCurrentCAAResearchWitness.v1','status':'PASS_CURRENT_CAA_STAGE20_24','product_authority_claimed':False,'main_code_sha':'d3c4f39ddecf2dc73e7c532ec352fc97ad92579b','candidate_lineage_hash':candidate.candidate_lineage_hash,'visual_mesh_set_hash':visual.set_hash,'appearance_domain_hash':domain.domain_hash,'static_qualification_hash':static.qualification_hash,'mechanical_carrier_evidence_hash':carrier.carrier_evidence_hash,'appearance_asset_hash':asset.asset_hash,'source_owned_visual_mesh_mode':bool(dict(asset.metadata or {}).get('source_owned_visual_mesh_mode')),'cross_view_completion_used':bool(dict(asset.metadata or {}).get('cross_view_completion_used')),'generated_appearance_used':bool(dict(asset.metadata or {}).get('generated_appearance_used'))}
    (child/'CAA_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('CAA_PASS',json.dumps(receipt,sort_keys=True),flush=True)

if __name__=='__main__': main()
