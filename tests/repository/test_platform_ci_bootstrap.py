import json

from tools import platform_ci_bootstrap as bootstrap


def test_fingerprint_changes_with_requirements():
    assert bootstrap.fingerprint(["pytest"]) != bootstrap.fingerprint(["pytest", "Pillow==11.3.0"])


def test_fingerprint_changes_with_requirement_file_bytes(tmp_path):
    req = tmp_path / "deps.txt"
    req.write_text("pytest==8.4.2\n", encoding="utf-8")
    first = bootstrap.fingerprint([], [req])
    req.write_text("pytest==8.4.1\n", encoding="utf-8")
    second = bootstrap.fingerprint([], [req])
    assert first != second


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


def test_existing_requirement_file_environment_is_reused_without_pip(tmp_path, monkeypatch):
    req = tmp_path / "deps.txt"
    req.write_text("pytest==8.4.2\n", encoding="utf-8")
    digest = bootstrap.fingerprint([], [req])
    target = tmp_path / "venvs" / digest
    (target / "bin").mkdir(parents=True)
    (target / "bin" / "python").write_text("stub")
    (target / ".realsas-env.json").write_text(json.dumps({"fingerprint": digest}))

    def must_not_run(*args, **kwargs):
        raise AssertionError("pip must not run")

    monkeypatch.setattr(bootstrap.subprocess, "run", must_not_run)
    report = bootstrap.ensure_venv(tmp_path, [], requirement_files=[req])
    assert report["status"] == "CACHE_HIT"
    assert report["fingerprint"] == digest
