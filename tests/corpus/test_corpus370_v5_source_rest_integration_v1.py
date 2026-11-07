"""V5 source-rest input identity, forbidden promotions and main-deploy guard."""
import json
from pathlib import Path

import pytest

from tools.corpus.audit_corpus370_v5_source_rest_courts_v1 import row_input
from tools.truth_foundry.platform_corpus370_source_surface_operator_v3 import prepare
from test_corpus370_platform_family_input_bundle_v3 import fixture, sha


def one_v4_row(tmp_path):
    _,_,_,_,_,packet=fixture(tmp_path)
    names={
        'import':'imported.json',
        'admission':'IRIS_OBSERVATION_ADMISSION.json',
        'candidate':'SOURCE_REST_RENDER_CANDIDATE.json',
        'snapshot':'SOURCE_SCENE_SNAPSHOT.json',
        'camera':'camera_bundle.json',
    }
    keys={
        'import':'admitted_import_geometry',
        'admission':'observation_admission',
        'candidate':'source_rest_candidate',
        'snapshot':'source_scene_snapshot',
        'camera':'camera_bundle',
    }
    refs={keys[k]:{'path':str(packet/v),'sha256':sha(packet/v)} for k,v in names.items()}
    return {
        'subject_id':'subject_000',
        'source_sha256':'a'*64,
        'evidence_producer':'NEW_V4_REPAIR_FROM_ORIGINAL_FAIL',
        'parent_failure_receipt':{'path':str(tmp_path/'old-fail.json'),'sha256':'b'*64},
        'packet_path':str(packet),
        'bindings':refs,
    },packet


def test_v4_source_surface_input_ref_is_exact_original_admitted_import(tmp_path):
    row,packet=one_v4_row(tmp_path)
    paths=row_input(row)
    assert paths['import']['sha256']==sha(packet/'imported.json')
    assert paths['candidate']['sha256']==sha(packet/'SOURCE_REST_RENDER_CANDIDATE.json')


@pytest.mark.parametrize('tamper',['import_bytes','camera_bytes','redirect_import','missing_old_failure'])
def test_v4_cannot_court_fake_source_surface_input(tmp_path,tamper):
    row,packet=one_v4_row(tmp_path)
    if tamper=='import_bytes':
        p=packet/'imported.json';p.write_bytes(p.read_bytes()+b' ')
    elif tamper=='camera_bytes':
        p=packet/'camera_bundle.json';p.write_bytes(p.read_bytes()+b' ')
    elif tamper=='redirect_import':
        import_path=tmp_path/'cloned-imported.json'
        import_path.write_bytes((packet/'imported.json').read_bytes())
        row['bindings']['admitted_import_geometry']['path']=str(import_path)
    else:
        row.pop('parent_failure_receipt')
    with pytest.raises(ValueError,match='SOURCE_COURTS_V5_'):
        row_input(row)


def test_operator_refuses_old_schema_and_undeployed_main(tmp_path):
    request={'schema':'RealSaS.Corpus370SourceSurfaceTruthOperatorRequest.v3',
             'enabled':True,'training_allowed':False}
    platform=tmp_path/'platform'
    platform.mkdir()
    (platform/'deployment.json').write_text(json.dumps({'code_sha':'1'*40}))
    kwargs=dict(repo=Path.cwd(),platform_root=platform,
                authority_root=tmp_path,expected_sha='0'*40,
                output=tmp_path/'out',api_root='http://127.0.0.1:1',
                rclone_remote='none',submit=False)
    with pytest.raises(ValueError,match='DEPLOYED_MAIN_DRIFT'):
        prepare(request,**kwargs)
    request['schema']='RealSaS.Corpus370ObservationTruthOperatorRequest.v2'
    with pytest.raises(ValueError,match='CORPUS370_TRUTH_OPERATOR_SCOPE'):
        prepare(request,**kwargs)


def test_finite_v3_plan_has_no_fabricated_sdf_sign_or_training():
    from tools.platform_truth_cohort_bootstrap_v1 import plan_manifest_keys
    p=json.loads(Path('canonical/CORPUS370_FAMILY_TRUTH_EXECUTION_PLAN_V3.json').read_text())
    stages={r['id']:r for r in p['stages']}
    assert 'source_surface' in plan_manifest_keys(p)
    assert 'presentation_truth' not in plan_manifest_keys(p)
    assert 'iris_fit' not in plan_manifest_keys(p)
    assert stages['50_SOURCE_SURFACE_ACQUIRED']['depends_on']!=[
        '51_SOURCE_CAPABILITIES_ACCOUNTED']
    assert '51_SOURCE_CAPABILITIES_ACCOUNTED' not in stages['50_SOURCE_SURFACE_ACQUIRED']['depends_on']
    assert p['corpus370_contract']['training_authorized'] is False
