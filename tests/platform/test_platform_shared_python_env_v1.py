"""Synthetic shared developer-service venv tests; no pip or real Platform writes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import platform_shared_python_env_v1 as m


def release(tmp_path: Path, name: str, *, numpy_pin: str = "numpy==2.1.3"):
    root = tmp_path / "home/monster/realsas_platform"
    folder = root / "releases" / name
    source = folder / "source"
    req = source / "requirements"
    req.mkdir(parents=True)
    (req / "platform-deployment.txt").write_text("-r mainline-ci.txt\npgserver==0.1.4\n")
    (req / "mainline-ci.txt").write_text(numpy_pin + "\n")
    return root, source, folder


def test_recursive_requirements_closure_and_identity(tmp_path):
    _, source, _ = release(tmp_path, "a" * 40 + "-11111111")
    actual = m.requirement_closure(source)
    assert [x.name for x in actual] == ["platform-deployment.txt", "mainline-ci.txt"]
    assert all(x.is_file() for x in actual)
    assert m.fingerprint([], actual) == m.fingerprint([], actual)


def test_modifying_included_requirement_rekeys_env(tmp_path):
    _, source1, _ = release(tmp_path, "a" * 40 + "-11111111")
    _, source2, _ = release(tmp_path, "b" * 40 + "-22222222", numpy_pin="numpy==2.1.4")
    assert m.fingerprint([], m.requirement_closure(source1)) != m.fingerprint([], m.requirement_closure(source2))


def test_symlinked_requirements_escape_is_rejected(tmp_path):
    _, source, _ = release(tmp_path, "a" * 40 + "-11111111")
    ext = tmp_path / "elsewhere.txt"
    ext.write_text("bad\n")
    (source / "requirements/mainline-ci.txt").unlink()
    (source / "requirements/mainline-ci.txt").symlink_to(ext)
    with pytest.raises(RuntimeError, match="OUTSIDE_PINNED"):
        m.requirement_closure(source)


def test_bad_recursive_include_is_rejected(tmp_path):
    _, source, _ = release(tmp_path, "a" * 40 + "-11111111")
    (source / "requirements/platform-deployment.txt").write_text("-r ../../secret.txt\n")
    with pytest.raises(RuntimeError, match="UNSUPPORTED_INCLUDE"):
        m.requirement_closure(source)


def test_two_releases_share_one_python_env(tmp_path, monkeypatch):
    root, s1, r1 = release(tmp_path, "a" * 40 + "-11111111")
    _, s2, r2 = release(tmp_path, "b" * 40 + "-22222222")
    created = []

    def bootstrap(cache_root, pkgs, *, requirement_files):
        digest = m.fingerprint(pkgs, requirement_files)
        dest = cache_root / "venvs" / digest
        if not dest.exists():
            (dest / "bin").mkdir(parents=True)
            (dest / "bin/python").write_text("mock")
            (dest / ".realsas-env.json").write_text(json.dumps({"fingerprint": digest}))
            created.append(dest)
        return {"venv": str(dest), "fingerprint": digest}

    monkeypatch.setattr(m, "ensure_venv", bootstrap)
    link1, digest1 = m.provision_shared_venv(root, s1, r1)
    link2, digest2 = m.provision_shared_venv(root, s2, r2)
    assert link1.is_symlink() and link2.is_symlink()
    assert link1.resolve() == link2.resolve()
    assert digest1 == digest2
    assert len(created) == 1


def test_dirty_shared_env_fails_closed_without_mutating(tmp_path, monkeypatch):
    root, source, folder = release(tmp_path, "a" * 40 + "-11111111")
    req = m.requirement_closure(source)
    digest = m.fingerprint([], req)
    dest = root / "shared_python_envs" / "venvs" / digest
    (dest / "bin").mkdir(parents=True)
    (dest / "bin/python").write_text("mock")
    (dest / ".realsas-env.json").write_text('{"fingerprint":"drift"}')

    def forbidden(*args, **kwargs):
        pytest.fail("mutating bootstrap must not run after invalid stamp")
    monkeypatch.setattr(m, "ensure_venv", forbidden)
    with pytest.raises(RuntimeError, match="IDENTITY_DRIFT_NO_REBUILD"):
        m.provision_shared_venv(root, source, folder)
    assert not (folder / "venv").exists()


def test_shared_env_wrong_root_refused(tmp_path):
    root, source, folder = release(tmp_path, "a" * 40 + "-11111111")
    with pytest.raises(RuntimeError, match="RELEASE_SCOPE_DRIFT"):
        m.provision_shared_venv(root, source, root)


def test_real_python_venv_symlink_imports_shared_site_packages(tmp_path, monkeypatch):
    """Integration of actual stdlib venv with release/venv path; zero network."""
    import subprocess
    import venv

    root, source, folder = release(tmp_path, "a" * 40 + "-11111111")

    def bootstrap(cache_root, packages, *, requirement_files):
        digest = m.fingerprint(packages, requirement_files)
        dest = cache_root / "venvs" / digest
        venv.EnvBuilder(with_pip=False).create(dest)
        package_dir = subprocess.run(
            [str(dest / "bin/python"), "-c",
             "import site; print(site.getsitepackages()[0])"],
            check=True, capture_output=True, text=True
        ).stdout.strip()
        (Path(package_dir) / "realsas_shared_probe.py").write_text(
            "SOURCE='CACHED_IMMUTABLE_ENV'\n")
        (dest / ".realsas-env.json").write_text(json.dumps({"fingerprint": digest}))
        return {"venv": str(dest), "fingerprint": digest}

    monkeypatch.setattr(m, "ensure_venv", bootstrap)
    link, _ = m.provision_shared_venv(root, source, folder)
    completed = subprocess.run(
        [str(link / "bin/python"), "-c",
         "import realsas_shared_probe; print(realsas_shared_probe.SOURCE)"],
        check=True, capture_output=True, text=True
    )
    assert completed.stdout.strip() == "CACHED_IMMUTABLE_ENV"
