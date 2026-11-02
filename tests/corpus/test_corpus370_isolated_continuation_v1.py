import copy
import hashlib
import json
import struct
from pathlib import Path

import pytest

from tools.truth_foundry.continue_corpus370_isolated_truth_v1 import repair_dependencies
from tools.corpus.inspect_corpus370_closure_inputs_v1 import superseded_campaigns


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    source = tmp_path / 'raw' / 'assets' / 'subject.glb'
    source.parent.mkdir(parents=True)
    doc = json.dumps({'asset': {'version': '2.0'}, 'images': [{'uri': 'skin%20texture.png'}]}).encode()
    doc += b' ' * (-len(doc) % 4)
    source.write_bytes(struct.pack('<III', 0x46546C67, 2, 20 + len(doc)) + struct.pack('<II', len(doc), 0x4E4F534A) + doc)
    texture = source.parent / 'skin texture.png'
    texture.write_bytes(b'original-source-texture')
    prepared = tmp_path / 'old' / source.name
    prepared.parent.mkdir()
    prepared.write_bytes(source.read_bytes())
    sections = {'source_files': [{'role': 'primary_source', 'source_kind': 'PRIMARY', 'path': str(prepared),
        'expected_sha256': sha(source), 'expected_size_bytes': source.stat().st_size}],
        'observation': {'path': '/immutable/observation'}, 'source_license': {'license': 'CC0'}}
    row = {'source_relative_path': 'assets/subject.glb', 'source_sha256': sha(source)}
    index = {'assets/skin texture.png': {'sha256': sha(texture), 'size_bytes': texture.stat().st_size}}
    return sections, row, source.parent.parent, index, tmp_path / 'repair'


def test_repair_is_new_exact_byte_closure_and_leaves_other_inputs_unchanged(tmp_path):
    args = fixture(tmp_path)
    old = copy.deepcopy(args[0])
    primary = Path(old['source_files'][0]['path'])
    before = primary.read_bytes()
    repaired, receipt = repair_dependencies(*args)
    assert args[0] == old and primary.read_bytes() == before
    assert list(primary.parent.iterdir()) == [primary]
    assert repaired['observation'] == old['observation']
    assert repaired['source_license'] == old['source_license']
    source, dependency = repaired['source_files']
    assert Path(source['path']).read_bytes() == before
    assert Path(dependency['path']).read_bytes() == b'original-source-texture'
    assert source['expected_sha256'] == old['source_files'][0]['expected_sha256']
    assert receipt['raw_source_mutated'] is False and receipt['training_authorized'] is False
    assert repair_dependencies(*args) == (repaired, receipt)
    assert repair_dependencies(repaired, *args[1:])[1]['status'] == 'UNCHANGED_SUBJECT_INPUT'


@pytest.mark.parametrize('field,value', [('sha256', '0' * 64), ('size_bytes', 1)])
def test_source_dependency_must_match_exact_manifest_bytes_and_size(tmp_path, field, value):
    args = fixture(tmp_path)
    args[3]['assets/skin texture.png'][field] = value
    with pytest.raises(RuntimeError, match='DEPENDENCY_SOURCE_DRIFT'):
        repair_dependencies(*args)


def test_supersession_requires_hash_bound_parent_and_exact_cohort(tmp_path):
    root = tmp_path / 'parent'
    root.mkdir()
    parent = {'campaign_id': 'old', 'member_count': 1, 'exact_source_roster_sha256': 'r'}
    p = root / 'PREPARATION.json'
    p.write_text(json.dumps(parent))
    entry = {'scope': str(root), 'PREPARATION.json': parent,
             'CAMPAIGN_BOOTSTRAP.json': {'run_bindings': [{'subject_id': 's'}]}}
    child = {'scope': str(tmp_path / 'child'), 'PREPARATION.json': {**parent, 'campaign_id': 'new',
        'parent_campaign_id': 'old', 'superseded_stage_ids': ['49_SUBJECT_TRUTH_FAMILIES'],
        'parent_preparation': {'path': str(p), 'sha256': sha(p)}},
        'CAMPAIGN_BOOTSTRAP.json': {'run_bindings': [{'subject_id': 's'}]}}
    assert superseded_campaigns([child, entry]) == {'old': 'new'}
    child['CAMPAIGN_BOOTSTRAP.json']['run_bindings'][0]['subject_id'] = 'other'
    with pytest.raises(ValueError, match='MEMBER_DRIFT'):
        superseded_campaigns([child, entry])
    child['CAMPAIGN_BOOTSTRAP.json']['run_bindings'][0]['subject_id'] = 's'
    p.write_text('{}')
    with pytest.raises(ValueError, match='SCOPE_DRIFT'):
        superseded_campaigns([child, entry])
