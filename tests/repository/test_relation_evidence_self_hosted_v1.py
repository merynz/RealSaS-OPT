from __future__ import annotations

from pathlib import Path


WORKFLOW = Path('.github/workflows/iris_vnext_relation_evidence_self_hosted_v1.yml')


def _source() -> str:
    return WORKFLOW.read_text(encoding='utf-8')


def test_relation_evidence_lane_is_self_hosted_and_artifact_only():
    source = _source()
    assert 'runs-on: [self-hosted, linux, x64, realsas]' in source
    assert 'actions/upload-artifact@v4' in source
    assert 'remote_writes=false' in source
    assert 'training_authorized=false' in source
    assert 'V9_EVIDENCE_SOURCE_SHA: 7029ee3f48787aa9f7e27eecef89155eb9368a96' in source


def test_relation_evidence_lane_cannot_start_training_or_write_remote():
    source = _source()
    forbidden = (
        'run_v9_focused_relation.py',
        'IRISVNextFocusedRelationTrainer',
        'optimizer.step',
        'torch.optim',
        'AUTHORIZE_IRIS_V9_FOCUSED_RELATION_OPTIMIZER_STEPS',
        'rclone sync',
        'rclone move',
        'rclone delete',
        'rclone purge',
    )
    assert all(token not in source for token in forbidden)


def test_relation_evidence_lane_uses_remote_only_as_read_source():
    source = _source()
    assert 'rclone copy "$V9_CANDIDATE_REMOTE/teachers" "$RUN_ROOT/teachers"' in source
    assert 'rclone copy "$V9_CANDIDATE_REMOTE/receipts" "$RUN_ROOT/v9_receipts"' in source
    # copyid is Drive file-id -> local destination only in this workflow.
    assert 'rclone backend copyid gdrive: "$STAGE01_FILE_ID" "$stage_zip"' in source
    assert 'rclone backend copyid gdrive: "$file_id" "$dst"' in source
    assert 'gdrive:RealSaS_Fit8_IRIS_vNext_candidates/${{ github.sha }}' not in source


def test_relation_evidence_lane_requires_exact_sealed_teacher_byte_court():
    source = _source()
    assert "receipt.get('status') != 'ROUNDTRIP_VERIFIED_CANDIDATE'" in source
    assert "receipt.get('file_count') != 57" in source
    assert "row['path'].startswith('teachers/')" in source
    assert 'len(rows) != 17' in source
    assert "sha(path) != row['sha256']" in source
