from __future__ import annotations

"""Main-owned BUSTED website and proof-gated current Studio API boundary.

These tests prove routing, source preservation and public workspace isolation.
They never claim an RSS.vNext learned checkpoint or a qualified public compile.
"""
from dataclasses import replace
from hashlib import sha256
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile
import json
from pathlib import Path
from threading import Thread
from http.server import ThreadingHTTPServer
from urllib import error, request

import pytest

from product.busted_site.server import BustedApplication
from product.living_compile.common import LivingCompileError
from product.living_compile.server import make_handler
from tests.product.test_living_compile_v4 import fixture as legacy_fixture

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "product" / "busted_site"


def test_pages_and_studio_use_exact_main_owned_static_assets():
    app = BustedApplication(SITE)
    for path, expected in (
        ("/", b"BUSTED"),
        ("/index.html", b"BUSTED"),
        ("/studio.html", b"BUSTED"),
        ("/motion.html", b"html"),
        ("/export.html", b"html"),
        ("/static/js/awaken.js", b"api_gateway_v1"),
        ("/static/js/app.js", b"api_gateway_v1"),
        ("/static/js/api_gateway_v1.js", b"BUSTED_API_ORIGIN"),
        ("/runtime-config.js", b"__BUSTED_PUBLIC_API_ORIGIN__"),
    ):
        code, payload, ctype = app.dispatch("GET", path)
        assert code == 200
        assert expected in payload
        assert ctype in ("text/html", "text/javascript", "application/javascript", "application/octet-stream")
    with pytest.raises(LivingCompileError):
        app.dispatch("GET", "/static/../../etc/passwd")


def test_frontend_calls_public_configurable_api_not_pages_fake_compiler():
    app = BustedApplication(SITE)
    _, p, _ = app.dispatch("GET", "/api/project")
    assert p["product_name"] == "BUSTED"
    assert p["website_owner"] == "MAIN_PRODUCT_BUSTED_SITE"
    assert not p["compile_available"]
    assert not p["public_compile_gateway"]
    assert p["input_policy"] == "TWO_OBSERVED_VIEWS_FRONT_BACK"
    for html_name in ("index.html", "studio.html"):
        text = (SITE / html_name).read_text("utf8")
        assert '<script src="./runtime-config.js"></script>' in text
    for script in ("awaken.js", "app.js", "product_shell.js"):
        source = (SITE / "static/js" / script).read_text("utf8")
        assert "api_gateway_v1.js" in source
        assert "apiFetch(" in source
    with pytest.raises(LivingCompileError, match="compile-from-UI"):
        app.dispatch("POST", "/api/compile/start", {"views": []})
    awaken = (SITE / "static/js/awaken.js").read_text("utf8")
    assert "RSS_VNEXT_REQUIRED_SOURCE_VIEWS = 2" in awaken
    assert "selectedFiles.length !== RSS_VNEXT_REQUIRED_SOURCE_VIEWS" in awaken
    assert "RSS_VNEXT_TWO_OBSERVED_VIEWS_FRONT_AND_BACK_REQUIRED" in awaken
    assert "slot: index === 0 ? 'front' : 'back'" in awaken



def test_public_workspace_only_accepts_opaque_ids_under_declared_root(tmp_path: Path):
    token = "aBcDeF_1234567890"
    workspace = tmp_path / token
    workspace.mkdir()
    legacy_fixture(workspace)
    app = BustedApplication(SITE, tmp_path, "https://merynz.github.io", True)
    status, data, _ = app.dispatch("GET", f"/api/living/scene?bundle={token}")
    assert status == 200
    assert data["bundle_root"] == token
    assert data["counts"]["views"] == 8
    _, clips, _ = app.dispatch("GET", f"/api/runtime/clips?bundle={token}")
    assert clips["clips"][0]["clip_id"] == "IDLE"
    for text in ("../../etc/passwd", str(workspace), "../aBcDeF_1234567890", "unknownbundletoken999"):
        with pytest.raises(LivingCompileError):
            app.dispatch("GET", "/api/living/scene?bundle=" + text)
    # Public code may not use raw paths to read arbitrary operator assets.
    with pytest.raises(LivingCompileError):
        app.dispatch("GET", "/api/blob?bundle=../../etc&path=etc/passwd")
    _, project, _ = app.dispatch("GET", "/api/project")
    assert not project["compile_available"]


def test_public_gateway_requires_https_origin_and_workspace_root(monkeypatch, tmp_path):
    monkeypatch.setenv("BUSTED_PUBLIC_MODE", "1")
    monkeypatch.setenv("BUSTED_PUBLIC_ORIGIN", "http://untrusted.example")
    monkeypatch.setenv("BUSTED_WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(LivingCompileError, match="HTTPS_ORIGIN_REQUIRED"):
        BustedApplication.default()
    monkeypatch.setenv("BUSTED_PUBLIC_ORIGIN", "https://merynz.github.io")
    app = BustedApplication.default()
    assert app.public_mode is True
    assert app.workspace_root == tmp_path
    monkeypatch.delenv("BUSTED_WORKSPACE_ROOT")
    with pytest.raises(LivingCompileError, match="WORKSPACE_ROOT_REQUIRED"):
        BustedApplication.default()


def test_cross_origin_preflight_only_for_exact_authorized_pages_origin(tmp_path):
    app = BustedApplication(SITE, tmp_path, "https://merynz.github.io", True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        req = request.Request(
            base + "/api/compile/start", method="OPTIONS",
            headers={"Origin": "https://merynz.github.io",
                     "Access-Control-Request-Method": "POST",
                     "Access-Control-Request-Headers": "content-type"},
        )
        with request.urlopen(req, timeout=3) as result:
            assert result.status == 204
            assert result.headers["Access-Control-Allow-Origin"] == "https://merynz.github.io"
            assert "POST" in result.headers["Access-Control-Allow-Methods"]

        bad = request.Request(base + "/api/project", method="OPTIONS",
                              headers={"Origin": "https://evil.example"})
        with pytest.raises(error.HTTPError) as rejected:
            request.urlopen(bad, timeout=3)
        assert rejected.value.code == 403

        get = request.Request(base + "/api/project", headers={"Origin": "https://merynz.github.io"})
        with request.urlopen(get, timeout=3) as result:
            assert result.headers["Access-Control-Allow-Origin"] == "https://merynz.github.io"
            assert json.load(result)["compile_available"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_public_native_rss_delivery_is_proof_gated_scoped_and_byte_verified(tmp_path):
    token = "QualifiedNative123456"
    workspace = tmp_path / token
    workspace.mkdir()
    legacy_fixture(workspace)
    native = workspace / "native" / "realsas_runtime.rss"
    native.parent.mkdir()
    with ZipFile(native, "w") as archive:
        archive.writestr("manifest.txt", "real-native-test-fixture\n")
        archive.writestr("motion.bin", b"provenance-aware-fixture")
    raw = native.read_bytes()
    app = BustedApplication(SITE, tmp_path, "https://merynz.github.io", True)
    code, receipt, mime = app.dispatch("POST", "/api/export", {"bundle": token})
    assert code == 200 and mime == "application/json"
    assert receipt["schema_version"] == "RealSaS.BustedQualifiedNativeDownload.v1"
    assert receipt["authority"] == "EXISTING_PROOF_GATED_NATIVE_RUNTIME"
    assert receipt["content_sha256"] == sha256(raw).hexdigest()
    assert receipt["size_bytes"] == len(raw)
    assert receipt["edited_studio_state_included"] is False
    assert str(workspace) not in json.dumps(receipt)
    parsed = urlsplit(receipt["download_url"])
    assert parsed.path == "/api/blob"
    assert parse_qs(parsed.query) == {
        "bundle": [token], "path": ["native/realsas_runtime.rss"],
    }
    got_status, data, got_mime = app.dispatch("GET", receipt["download_url"])
    assert got_status == 200 and data == raw
    assert got_mime == "application/octet-stream"

    proof_path = workspace / "proof" / "product_proof_bundle_ir.json"
    proof = json.loads(proof_path.read_text())
    proof["overall_status"] = "FAIL"
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(LivingCompileError, match="PROOF_PASS_REQUIRED"):
        app.dispatch("POST", "/api/export", {"bundle": token})
    with pytest.raises(LivingCompileError, match="PROOF_PASS_REQUIRED"):
        app.dispatch("GET", receipt["download_url"])
    with pytest.raises(LivingCompileError):
        app.dispatch("GET", "/api/blob?bundle=../QualifiedNative123456&path=native/realsas_runtime.rss")


def test_native_export_rejects_corrupt_archive_and_outside_symlink(tmp_path):
    token = "QualifiedNative123456"
    workspace = tmp_path / token
    workspace.mkdir()
    legacy_fixture(workspace)
    native = workspace / "native" / "realsas_runtime.rss"
    native.parent.mkdir()
    native.write_bytes(b"not-a-valid-native-rss")
    app = BustedApplication(SITE, tmp_path, "https://merynz.github.io", True)
    with pytest.raises(LivingCompileError, match="RSS_DECODE_INVALID"):
        app.dispatch("POST", "/api/export", {"bundle": token})
    native.unlink()
    outside = tmp_path / "elsewhere.rss"
    outside.write_bytes(b"exfiltration")
    native.symlink_to(outside)
    with pytest.raises(LivingCompileError, match="path escapes bundle root"):
        app.dispatch("POST", "/api/export", {"bundle": token})
