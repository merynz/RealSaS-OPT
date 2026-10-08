import json

from tools import platform_ci_bootstrap as bootstrap


def test_fingerprint_changes_with_requirements():
    assert bootstrap.fingerprint(["pytest"]) != bootstrap.fingerprint(["pytest", "Pillow==11.3.0"])


def test_existing_exact_environment_is_reused_without_pip(tmp_path, monkeypatch):
    requirements = ["pytest"]
    digest = bootstrap.fingerprint(requirements)
    target = tmp_path / "venvs" / digest
    (target / "bin").mkdir(parents=True)
    (target / "bin" / "python").write_text("stub")
    (target / ".realsas-env.json").write_text(json.dumps({"fingerprint": digest}))

    def must_not_run(*args, **kwargs):
        raise AssertionError("pip must not run")

    monkeypatch.setattr(bootstrap.subprocess, "run", must_not_run)
    report = bootstrap.ensure_venv(tmp_path, requirements)
    assert report["status"] == "CACHE_HIT"
    assert report["fingerprint"] == digest
