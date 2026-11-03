import json,subprocess,sys
from pathlib import Path
import pytest
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.orchestrator.adapters import source_surface_v1 as surface
from compiler.realsas_compiler_services.orchestrator.adapters import source_capabilities_v1 as capabilities
from compiler.realsas_compiler_services.orchestrator.adapters import corpus_family_v2 as families
from compiler.realsas_compiler_services.orchestrator.adapters.measurement_io_v1 import write,ref
from tools.corpus.validate_source_capability_absence_v1 import validate as absence
from test_source_rest_geometry_v1 import fixture


def test_v3_graph_does_not_make_surface_depend_on_calibration():
    plan=json.loads(Path('canonical/CORPUS370_FAMILY_TRUTH_EXECUTION_PLAN_V3.json').read_text())
    mainline.validate_plan(plan,require_product_pass_authority=False)
    stages={s['id']:s for s in plan['stages']}
    assert '51_SOURCE_CAPABILITIES_ACCOUNTED' not in stages['50_SOURCE_SURFACE_ACQUIRED']['depends_on']
    assert '47_SOURCE_RESPONSE_CALIBRATION' not in stages
    assert stages['49_SUBJECT_TRUTH_FAMILIES']['adapter'].endswith('corpus_family_v2:produce_subject_family_stage')


def context(root,manifest,stage_id):
    return {'run_root':root,'run_manifest':manifest,'stage':{'id':stage_id,'policy':{'product_pass_authority':False}},'ledger':{'stages':[]}}


def test_source_surface_survives_known_calibration_failure_end_to_end(tmp_path,monkeypatch):
    ip,ap,cp,sp=fixture(tmp_path)
    camera=write(tmp_path/'camera.json',{'schema':'Synthetic.Camera.v1'})
    manifest={'subject_id':'subject','source_files':[{'source_kind':'PRIMARY','expected_sha256':'a'*64}],
        'source_surface':{'mode':'EXACT_ADMITTED_IMPORT_SURFACE_ONLY','admitted_import_geometry':ref(ip)},
        'source_audit':{'iris_observation_admission':ref(ap),'source_rest_candidate':ref(cp),'source_scene_snapshot':ref(sp),'matrix_tolerance':1e-6},
        'observation':{'camera_bundle':ref(camera)}}
    monkeypatch.setattr(surface,'stage_output_payload',lambda ctx,stage,schema:{'files':[{'sha256':'a'*64}]} if stage.startswith('01') else {})
    first=context(tmp_path/'run',manifest,'50_SOURCE_SURFACE_ACQUIRED');result=surface.produce_source_surface_stage(first)
    rest=next(r for r in result['outputs'] if r['schema']=='RealSaS.AdmittedSourceRestGeometry.v1')
    monkeypatch.setattr(capabilities,'stage_output_payload',lambda *a:json.loads(Path(rest['path']).read_text()))
    def failed(_):raise QualificationError('TEACHER_SCENE_NO_SUPPORTED_KINEMATIC_CAPABILITY')
    monkeypatch.setattr(capabilities,'measure_teacher_scene_stage',failed)
    second=context(tmp_path/'run',manifest,'51_SOURCE_CAPABILITIES_ACCOUNTED');record=capabilities.measure_source_capabilities_stage(second)
    cap=next(r for r in record['outputs'] if r['schema']=='RealSaS.SourceCapabilityMeasurement.v1')
    cap_value=json.loads(Path(cap['path']).read_text())
    assert cap_value['teacher_scene'] is None and cap_value['target_arrays_issued'] is False
    proof=absence(cap['path'],subject_id='subject',source_sha256='a'*64)
    assert proof['authority']['qualified_target_count']==0
    third=context(tmp_path/'run',manifest,'49_SUBJECT_TRUTH_FAMILIES')
    third['ledger']['stages']=[{'id':first['stage']['id'],'outputs':result['outputs']},{'id':second['stage']['id'],'outputs':record['outputs']}]
    monkeypatch.setattr(families,'stage_output_payload',lambda ctx,stage,schema:json.loads(Path(rest['path'] if stage.startswith('50') else cap['path']).read_text()))
    final=families.produce_subject_family_stage(third)
    index=next(r for r in final['outputs'] if r['schema']=='RealSaS.Corpus370FamilyEvidenceIndex.v1')
    states={f:c['status'] for f,c in json.loads(Path(index['path']).read_text())['subjects'][0]['families'].items()}
    assert states=={'SURFACE_GEOMETRY_UDF':'PASS','SURFACE_NORMALS':'PASS','MECHANICAL_SURFACE_TOPOLOGY':'MASKED','CAUSAL_RESPONSE':'MASKED','TESSA_TEACHER':'MASKED'}
    assert 'measurement_failure' in cap_value and cap_value['measurement_failure']['reason'].endswith('NO_SUPPORTED_KINEMATIC_CAPABILITY')


@pytest.mark.parametrize('exception',[MemoryError('resource failure'),RuntimeError('producer bug')])
def test_untyped_or_resource_failure_is_not_a_scientific_mask(tmp_path,monkeypatch,exception):
    monkeypatch.setattr(capabilities,'stage_output_payload',lambda *a:{'source_sha256':'a'*64})
    def fail(_):raise exception
    monkeypatch.setattr(capabilities,'measure_teacher_scene_stage',fail)
    with pytest.raises(type(exception)):
        capabilities.measure_source_capabilities_stage(context(tmp_path,{'subject_id':'s'},'51_SOURCE_CAPABILITIES_ACCOUNTED'))


def test_surface_import_closure_does_not_load_a_teacher_or_torch():
    code='''import importlib.abc,sys
class Ban(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path,target=None):
  if fullname=='torch' or fullname.startswith('tools.training') or fullname.startswith('tools.truth_foundry.blender_teacher'):
   raise RuntimeError('SOURCE_SURFACE_MUST_BE_INDEPENDENT:'+fullname)
sys.meta_path.insert(0,Ban())
from compiler.realsas_compiler_services.orchestrator.adapters import source_surface_v1
'''
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
