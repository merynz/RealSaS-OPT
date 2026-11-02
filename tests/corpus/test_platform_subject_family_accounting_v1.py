from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

from compiler.realsas_compiler_services.orchestrator.adapters import corpus_family_v1 as family


def test_static_family_courts_import_without_neural_runtime():
    program = '''
import importlib.abc, sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("STATIC_COURT_IMPORTED_TORCH")
sys.meta_path.insert(0, NoTorch())
from compiler.realsas_compiler_services.orchestrator.adapters import corpus_family_v1
assert "torch" not in sys.modules
'''
    subprocess.run([sys.executable, '-c', program], check=True,
                   cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)


def test_geometry_and_response_failures_preserve_independent_mechanical_and_tessa(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise ValueError('FALSIFICATION_EXPECTED_FAILURE')
    monkeypatch.setattr(family.geometry, 'build', fail)
    monkeypatch.setattr(family.response, 'build', fail)

    def mechanical(sid, geometry_path, out):
        family.write(out / 'MECHANICAL_SURFACE_TOPOLOGY_MANIFEST_V1.json',
                     {'schema': 'Synthetic.Mechanical.v1'})
    monkeypatch.setattr(family.mechanical, 'build', mechanical)
    monkeypatch.setattr(family.mechanical_court, 'validate', lambda p: {
        'schema': 'Synthetic.MechanicalCourt.v1', 'status': 'PASS_INDEPENDENT_COURT',
        'family_candidate_outcomes': {'SURFACE_TOPOLOGY': 'QUALIFIED_CORE_CANDIDATE'},
        'authority': {'training_authorized': False}})

    def tessa(**kwargs):
        family.write(kwargs['output_dir'] / 'CORPUS_MECHANICAL_TESSA_TEACHER_MANIFEST_V5.json',
                     {'schema': 'Synthetic.Tessa.v1'})
    monkeypatch.setattr(family.tessa, 'build', tessa)
    monkeypatch.setattr(family.tessa_court, 'validate', lambda m: {
        'schema': 'Synthetic.TessaCourt.v1', 'status': 'PASS_INDEPENDENT_CONSUMER_COURT',
        'authority': {'training_authorized': False,
                      'tessa_mechanical_surface_consumer_court_passed': True}})
    scene = family.write(tmp_path / 'scene.json', {'schema': 'Synthetic.Scene.v1'})
    replay = family.write(tmp_path / 'replay.json', {'schema': 'Synthetic.Replay.v1'})
    row = family.measure_subject_families(sid='s', source_sha='1'*64,
        scene_path=scene, replay_path=replay, geometry_path=tmp_path / 'unused',
        snapshot_path=tmp_path / 'unused', camera_path=tmp_path / 'unused', root=tmp_path / 'out')
    states = {f: c['status'] for f, c in row['families'].items()}
    assert states == {'SURFACE_GEOMETRY_UDF': 'FAIL', 'SURFACE_NORMALS': 'FAIL',
                      'MECHANICAL_SURFACE_TOPOLOGY': 'PASS', 'CAUSAL_RESPONSE': 'FAIL',
                      'TESSA_TEACHER': 'PASS'}
    assert 'GLOBAL_SOLID_SDF' not in row['families']
    for claim in row['families'].values():
        path = Path(claim['court']['path'])
        assert path.is_file() and family.sha256_file(path) == claim['court']['sha256']


def test_released_family_target_skips_unrelated_presentation_measurement():
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / 'canonical/CORPUS370_FAMILY_TRUTH_EXECUTION_PLAN_V2.json').read_text())
    stages = {s['id']: s for s in plan['stages']}
    found = set()
    def visit(sid):
        if sid in found:
            return
        found.add(sid)
        for dep in stages[sid]['depends_on']:
            visit(dep)
    visit('49_SUBJECT_TRUTH_FAMILIES')
    assert '47_SOURCE_RESPONSE_CALIBRATION' in found
    assert '48_PRESENTATION_RESPONSE_TRUTH' not in found
    assert not any('TRAIN' in sid for sid in found)
