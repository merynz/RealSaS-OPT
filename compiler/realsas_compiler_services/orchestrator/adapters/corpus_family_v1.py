"""Platform-owned, failure-preserving subject-family measurement accounting.

Stage PASS means every requested family has an explicit outcome. Family PASS is
issued only by its independent numeric/consumer court; failed families remain FAIL.
This stage cannot authorize training or promote component sign to global sign.
"""
from __future__ import annotations

import json
from pathlib import Path
import traceback

from .adapter_io import load_file_ref, sha256_file, stage_output_payload
from .truth_foundry_v1 import _artifact, _require
from tools.corpus import build_component_geometry_authority_v1 as geometry
from tools.corpus import validate_component_geometry_authority_v1 as geometry_court
from tools.corpus import build_mechanical_surface_topology_manifest_v1 as mechanical
from tools.corpus import validate_mechanical_surface_topology_authority_v1 as mechanical_court
from tools.corpus import build_causal_response_authority_v2 as response
from tools.corpus import validate_causal_response_authority_v2 as response_court
from tools.training import build_corpus_mechanical_tessa_teacher_v5 as tessa
from tools.training import validate_corpus_mechanical_tessa_teacher_v5 as tessa_court


def ref(path: Path, schema=None):
    row = {'path': str(path.resolve()), 'sha256': sha256_file(path)}
    if schema:
        row['schema'] = schema
    return row


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    return path


def measure_subject_families(*, sid, source_sha, scene_path, replay_path,
                             geometry_path, snapshot_path, camera_path, root):
    """Run independent families even when another family fails its court."""
    families = {}
    paths = {}
    outcome_keys = {'SURFACE_GEOMETRY_UDF': 'SURFACE_GEOMETRY',
                    'SURFACE_NORMALS': 'ORIENTED_NORMALS',
                    'MECHANICAL_SURFACE_TOPOLOGY': 'SURFACE_TOPOLOGY',
                    'CAUSAL_RESPONSE': 'CAUSAL_RESPONSE'}

    def lane(name, family_names, produce, court, dependencies=()):
        path = root / name / 'COURT.json'
        try:
            _require(all(families[f]['status'] == 'PASS' for f in dependencies),
                     'CORPUS_FAMILY_REQUIRED_PARENT_NOT_PASS:' + name)
            candidate_path = produce()
            result = court(candidate_path)
            _require(str(result.get('status', '')).startswith('PASS_'),
                     'CORPUS_FAMILY_INDEPENDENT_COURT_NOT_PASS:' + name)
            _require((result.get('authority') or {}).get('training_authorized') is False,
                     'CORPUS_FAMILY_PREMATURE_TRAINING_AUTHORITY')
            for family in family_names:
                if family == 'TESSA_TEACHER':
                    _require((result.get('authority') or {}).get('tessa_mechanical_surface_consumer_court_passed') is True,
                             'CORPUS_FAMILY_TESSA_CONSUMER_NOT_QUALIFIED')
                else:
                    outcome = (result.get('family_candidate_outcomes') or {}).get(outcome_keys[family])
                    _require(outcome in {'QUALIFIED_CORE_CANDIDATE', 'QUALIFIED_MASKED_CANDIDATE',
                                         'QUALIFIED_CONSUMER_CANDIDATE', 'UNAVAILABLE_CANDIDATE'},
                             'CORPUS_FAMILY_COURT_OUTCOME_UNSUPPORTED:' + family)
            write(path, result)
            paths[name] = candidate_path
            claim = {'status': 'PASS', 'reason_codes': [],
                     'evidence_refs': [ref(candidate_path, json.loads(candidate_path.read_text())['schema'])],
                     'court': {**ref(path, result['schema']), 'status': 'PASS'}}
        except MemoryError:
            # Resource exhaustion is execution failure, never a scientific FAIL.
            raise
        except Exception as exc:
            failure = {'schema': 'RealSaS.CorpusFamilyExecutionFailure.v1',
                       'status': 'FAIL', 'subject_id': sid, 'source_sha256': source_sha,
                       'family_group': name, 'error_type': type(exc).__name__,
                       'reason': str(exc), 'traceback': traceback.format_exc(),
                       'training_authorized': False}
            write(path, failure)
            claim = {'status': 'FAIL', 'reason_codes': [str(exc).split(':')[0] or type(exc).__name__],
                     'evidence_refs': [ref(path, failure['schema'])],
                     'court': {**ref(path, failure['schema']), 'status': 'FAIL'}}
        for family in family_names:
            families[family] = dict(claim)
            if claim['status'] == 'PASS' and family != 'TESSA_TEACHER':
                if result['family_candidate_outcomes'][outcome_keys[family]] == 'UNAVAILABLE_CANDIDATE':
                    families[family].update(status='MASKED', reason_codes=['COURTED_NO_VALID_TARGET_ROWS'])
        print(json.dumps({'event': 'CORPUS_FAMILY_OUTCOME', 'subject_id': sid,
                          'families': list(family_names), 'status': claim['status'],
                          'reason_codes': claim['reason_codes'],
                          'family_status': {f: families[f]['status'] for f in family_names}}), flush=True)

    def produce_geometry():
        out = root / 'geometry'
        value = geometry.build(sid, geometry_path, snapshot_path, camera_path, out)
        return out / 'COMPONENT_GEOMETRY_AUTHORITY_MANIFEST_V1.json'

    lane('geometry', ('SURFACE_GEOMETRY_UDF', 'SURFACE_NORMALS'), produce_geometry,
         lambda p: geometry_court.validate(json.loads(p.read_text())))

    def produce_mechanical():
        out = root / 'mechanical'
        mechanical.build(sid, geometry_path, out)
        return out / 'MECHANICAL_SURFACE_TOPOLOGY_MANIFEST_V1.json'

    lane('mechanical', ('MECHANICAL_SURFACE_TOPOLOGY',), produce_mechanical,
         mechanical_court.validate)

    def produce_response():
        out = root / 'response'
        response.build(sid, paths['mechanical'], root / 'mechanical/COURT.json', scene_path, out)
        return out / 'CAUSAL_RESPONSE_AUTHORITY_MANIFEST_V2.json'

    lane('response', ('CAUSAL_RESPONSE',), produce_response, response_court.validate,
         ('MECHANICAL_SURFACE_TOPOLOGY',))

    chain = write(root / 'RESPONSE_BINDING.json', {
        'schema': 'RealSaS.PlatformBoundResponseInputs.v1', 'subject_id': sid,
        'inputs': {'admitted_observation_import': ref(replay_path)},
        'outputs': {'teacher_scene': ref(scene_path)}, 'training_authorized': False})

    def produce_tessa():
        out = root / 'tessa'
        tessa.build(subject_id=sid, mechanical_manifest_path=paths['mechanical'],
                    mechanical_court_path=root / 'mechanical/COURT.json',
                    response_chain_path=chain, camera_bundle_path=camera_path, output_dir=out)
        return out / 'CORPUS_MECHANICAL_TESSA_TEACHER_MANIFEST_V5.json'

    lane('tessa', ('TESSA_TEACHER',), produce_tessa,
         lambda p: tessa_court.validate(json.loads(p.read_text())),
         ('MECHANICAL_SURFACE_TOPOLOGY',))
    return {'subject_id': sid, 'source_sha256': source_sha, 'families': families}


def produce_subject_family_stage(ctx):
    _require(ctx['stage']['policy'].get('product_pass_authority') is False,
             'CORPUS_FAMILY_CANNOT_MINT_PRODUCT_AUTHORITY')
    manifest = ctx['run_manifest']
    sid = manifest['subject_id']
    scene = stage_output_payload(ctx, '47_SOURCE_RESPONSE_CALIBRATION', 'RealSaS.TeacherSceneIR.v2')
    upstream = next(row for row in ctx['ledger']['stages'] if row['id'] == '47_SOURCE_RESPONSE_CALIBRATION')
    scene_rows = [r for r in upstream['outputs'] if r['schema'] == 'RealSaS.TeacherSceneIR.v2']
    replay_rows = [r for r in upstream['outputs'] if r['schema'] == 'RealSaS.ImportedObservationGeometryReplay.v1']
    _require(len(scene_rows) == len(replay_rows) == 1, 'CORPUS_FAMILY_STAGE47_PARENT_CARDINALITY')
    scene_path = load_file_ref(scene_rows[0], json_required=False)
    replay_path = load_file_ref(replay_rows[0], json_required=False)
    geometry_rows = [r for r in scene['geometry_payloads'] if Path(r['path']).name == 'geometry_and_response.npz']
    _require(len(geometry_rows) == 1, 'CORPUS_FAMILY_GEOMETRY_PARENT_CARDINALITY')
    geometry_path = load_file_ref(geometry_rows[0], json_required=False)
    snapshot_path = load_file_ref(manifest['source_audit']['source_scene_snapshot'], json_required=False)
    camera_path = load_file_ref(manifest['observation']['camera_bundle'], json_required=False)
    source = [r for r in manifest['source_files'] if r.get('source_kind') == 'PRIMARY']
    _require(len(source) == 1, 'CORPUS_FAMILY_PRIMARY_SOURCE_CARDINALITY')
    source_sha = source[0]['expected_sha256']
    root = ctx['run_root'] / 'artifacts' / ctx['stage']['id']
    root.mkdir(parents=True, exist_ok=True)
    row = measure_subject_families(sid=sid, source_sha=source_sha, scene_path=scene_path,
        replay_path=replay_path, geometry_path=geometry_path, snapshot_path=snapshot_path,
        camera_path=camera_path, root=root)
    index = {'schema': 'RealSaS.Corpus370FamilyEvidenceIndex.v1',
             'status': 'SUBJECT_FAMILY_MEASUREMENT_ACCOUNTED__FINAL_ADMISSION_SEPARATE',
             'subject_count': 1, 'subjects': [row],
             'parents': {'teacher_scene': ref(scene_path), 'geometry': ref(geometry_path)},
             'component_sign_promoted_to_global': False, 'training_authorized': False}
    write(root / 'SUBJECT_FAMILY_EVIDENCE.json', index)
    outputs = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix not in {'.json', '.npz'}:
            continue
        schema = json.loads(path.read_text())['schema'] if path.suffix == '.json' else 'RealSaS.TruthFamilyNumericPayload.v1'
        outputs.append(_artifact(path, schema))
    return {'status': 'PASS', 'outputs': outputs,
            'diagnostics': {'accounting_complete': True,
                'family_status': {f: c['status'] for f, c in row['families'].items()},
                'training_authorized': False, 'product_authority_minted': False}}
