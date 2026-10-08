import io
from pathlib import Path
import tarfile

import pytest

from tools import platform_deploy as deploy
from tools.platform_deployment_smoke import smoke_plans
from tools import platform_deployment_smoke as smoke
from tools.platform_release_snapshot import snapshot
from compiler.realsas_compiler_services.platform_worker.stage_inputs import graph_node_sha256, released_execution_plan
from compiler.realsas_compiler_services.orchestrator import mainline


@pytest.mark.parametrize("value", ["/", str(Path.home()), ".", "/tmp", "/tmp/../tmp/unsafe", "/tmp/a b/unsafe"])
def test_deployment_rejects_broad_or_ambiguous_roots(value):
    with pytest.raises(RuntimeError, match="ROOT"):
        deploy.safe_root(value)


def test_symlinked_deployment_root_is_not_followed(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(RuntimeError, match="SYMLINKED"):
        deploy.safe_root(str(link))


def test_units_are_foreground_and_separate_persistent_data_from_code():
    root = Path("/home/monster/realsas_platform")
    units = deploy.service_units(root)
    assert len(units) == 7
    for value in units.values():
        assert "RUNNER_TRACKING_ID=" not in value
        assert "nohup" not in value and "sudo" not in value
        assert str(root / "current/source") not in value or "WorkingDirectory=" in value
    temporal = units["realsas-platform-temporal.service"]
    assert "--ip 127.0.0.1 --headless --db-filename /home/monster/realsas_platform/data/temporal.sqlite" in temporal
    assert "Restart=on-failure" in temporal
    assert "PartOf=realsas-platform.target" in temporal
    assert "Requires=realsas-platform-migrate.service" in units["realsas-platform-api.service"]
    assert "Type=oneshot" in units["realsas-platform-migrate.service"]
    assert " reset" not in "\n".join(units.values())


def test_temporal_hash_is_checked_before_any_extraction(tmp_path):
    destination = tmp_path / "temporal"
    with pytest.raises(RuntimeError, match="HASH_MISMATCH"):
        deploy.extract_temporal(b"untrusted archive", destination)
    assert not destination.exists()


def test_temporal_extracts_only_exact_regular_binary(tmp_path, monkeypatch):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        entry = tarfile.TarInfo("temporal")
        entry.size = 4
        archive.addfile(entry, io.BytesIO(b"test"))
        hostile = tarfile.TarInfo("../../do-not-extract")
        hostile.size = 6
        archive.addfile(hostile, io.BytesIO(b"hostile"))
    raw = data.getvalue()
    monkeypatch.setattr(deploy, "TEMPORAL_SHA256", deploy.hashlib.sha256(raw).hexdigest())
    destination = tmp_path / "temporal"
    deploy.extract_temporal(raw, destination)
    assert destination.read_bytes() == b"test"
    assert list(tmp_path.iterdir()) == [destination]


def test_missing_user_manager_stops_without_a_privilege_fallback(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return type("Result", (), {"returncode": 1})()
    monkeypatch.setattr(deploy.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="USER_SYSTEMD_UNAVAILABLE"):
        deploy.manager_environment()
    assert calls == [["systemctl", "--user", "show-environment"]]


def test_smoke_graph_removal_preserves_independent_component_identity():
    baseline, candidate = smoke_plans()
    manifest = {"source_license": {"source_pack": "test", "license_name": "test", "license_ref": "urn:test"}}
    old = snapshot(baseline, manifest, name="old", purpose="RESEARCH", created_by="ci")
    new = snapshot(candidate, manifest, name="new", purpose="RESEARCH", created_by="ci")
    first, second = old["manifest"]["stages"][1], new["manifest"]["stages"][0]
    assert first["ordinal"] == 2 and second["ordinal"] == 1
    for key in ("stage_id", "implementation_sha256", "policy_sha256", "semantic_parameters"):
        assert first[key] == second[key]
    assert graph_node_sha256(baseline["stages"][1]) == graph_node_sha256(candidate["stages"][0])
    for request, plan in ((old, baseline), (new, candidate)):
        actual = released_execution_plan(mainline.load_json(mainline.PLAN_PATH), request["manifest"]["dag"], mode="RESEARCH")
        assert mainline.content_sha256(actual) == mainline.content_sha256(plan)


def test_deployment_workflow_is_canonical_main_only_for_mutations():
    root = Path(__file__).resolve().parents[2]
    text = (root / ".github/workflows/platform_developer_deploy.yml").read_text()
    assert "runs-on: [self-hosted, linux, x64, realsas]" in text
    assert "if: github.event_name == 'push' && github.ref == 'refs/heads/main'" in text
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in text
    assert "platform_deploy.py" in text and "platform_deployment_smoke.py" in text


def test_smoke_registry_read_is_bound_to_actual_license_stage(tmp_path, monkeypatch):
    data = b'{}'
    digest = deploy.hashlib.sha256(data).hexdigest()
    storage_key = f'cas/sha256/{digest[:2]}/{digest[2:4]}/{digest}'
    destination = tmp_path / 'artifacts' / storage_key
    destination.parent.mkdir(parents=True)
    destination.write_bytes(data)
    attempt_id = '4e5af44d-7b74-4a24-a720-de1a4eb99844'
    seen = []
    def read(root, sql):
        seen.append(sql)
        return f'{attempt_id}|{digest}|{storage_key}'
    monkeypatch.setattr(smoke, 'registry_read', read)
    result = smoke.artifact_for_attempt(tmp_path, attempt_id, smoke_plans()[0]['stages'][1]['id'])
    assert result['content_sha256'] == digest
    assert "stage:02_SOURCE_LICENSE_PROVENANCE" in seen[0]
    with pytest.raises((ValueError, RuntimeError)):
        smoke.artifact_for_attempt(tmp_path, "invalid'", '02_SOURCE_LICENSE_PROVENANCE')
