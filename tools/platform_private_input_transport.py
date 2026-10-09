"""One-run RSA-OAEP transport for private download locators on a public CI repo.

The runner retains its private key locally and publishes only the public key.
Ciphertext is bound to the run ID and exact code SHA. Locators are never logged
or committed as plaintext; scientific bytes still require their pinned SHA256.

For larger handoffs, the encrypted payload may contain one hash-pinned manifest
locator instead of every transfer locator. The manifest is itself bound to the
same run ID/code SHA and then resolves to either the original transfer list or
one hash-pinned ZIP bundle whose inner files are independently verified.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone

MAX_TRANSFER_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_ENVELOPE_BYTES = 2 * 1024 * 1024
MAX_CONTENTS_API_BYTES = 4 * 1024 * 1024
MAX_INPUT_FILES = 128
SAFE_INPUT_NAME = re.compile(r"[A-Za-z0-9_.-]+")
ALLOWED_DOWNLOAD_HOST_SUFFIXES = (
    ".blob.core.windows.net",
    ".amazonaws.com",
    ".oaiusercontent.com",
)


def rsa(data, key, *, encrypt):
    with tempfile.TemporaryDirectory() as directory:
        source, target = Path(directory)/"in", Path(directory)/"out"
        source.write_bytes(data)
        command = ["openssl", "pkeyutl", "-encrypt" if encrypt else "-decrypt",
                   "-inkey", str(key), "-in", str(source), "-out", str(target),
                   "-pkeyopt", "rsa_padding_mode:oaep", "-pkeyopt", "rsa_oaep_md:sha256"]
        if encrypt:
            command.append("-pubin")
        subprocess.run(command, check=True, capture_output=True)
        return target.read_bytes()


def generate(directory, run_id, code_sha):
    directory.mkdir(parents=True, exist_ok=True)
    private = directory/"private.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048",
                    "-out", str(private)], check=True, capture_output=True)
    private.chmod(0o600)
    public = subprocess.check_output(["openssl", "pkey", "-in", str(private), "-pubout"]).decode()
    (directory/"public.json").write_text(json.dumps({"run_id": run_id, "code_sha": code_sha,
                                                   "public_key_pem": public}) + "\n")


def encrypt(public_path, payload_path, out):
    public = json.loads(public_path.read_text())
    payload = json.loads(payload_path.read_text())
    payload.update(run_id=public["run_id"], code_sha=public["code_sha"])
    raw = json.dumps(payload, separators=(",", ":")).encode()
    with tempfile.TemporaryDirectory() as directory:
        key = Path(directory)/"public.pem"
        key.write_text(public["public_key_pem"])
        chunks = [base64.b64encode(rsa(raw[i:i+190], key, encrypt=True)).decode()
                  for i in range(0, len(raw), 190)]
    out.write_text(json.dumps({"run_id": public["run_id"], "code_sha": public["code_sha"],
                              "public_key_sha256": hashlib.sha256(public["public_key_pem"].encode()).hexdigest(),
                              "plaintext_sha256": hashlib.sha256(raw).hexdigest(), "chunks": chunks}) + "\n")


def _require_identity(value, run_id, code_sha, error_code):
    if value.get("run_id") != run_id or value.get("code_sha") != code_sha:
        raise RuntimeError(error_code)


def _download_pinned(ref, label, max_bytes):
    if not isinstance(ref, dict):
        raise RuntimeError("PRIVATE_INPUT_REFERENCE_INVALID:" + label)
    uri = ref.get("download_url")
    size = ref.get("size_bytes")
    digest = ref.get("sha256")
    if (not isinstance(uri, str) or not isinstance(size, int) or size < 0 or size > max_bytes
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise RuntimeError("PRIVATE_INPUT_REFERENCE_INVALID:" + label)
    parsed = urllib.parse.urlsplit(uri)
    if (parsed.scheme != "https"
            or not (parsed.hostname or "").endswith(ALLOWED_DOWNLOAD_HOST_SUFFIXES)):
        raise RuntimeError("PRIVATE_INPUT_DOWNLOAD_HOST_INVALID")
    expiry = urllib.parse.parse_qs(parsed.query).get("se", [])
    if expiry:
        try:
            deadline = datetime.fromisoformat(expiry[0].replace("Z", "+00:00"))
        except ValueError:
            deadline = None
        if deadline is not None and deadline.tzinfo is not None and deadline <= datetime.now(timezone.utc):
            raise RuntimeError("PRIVATE_INPUT_DOWNLOAD_LOCATOR_EXPIRED:" + label)
    print("::add-mask::" + uri, flush=True)
    try:
        # The signed-download gateway rejects urllib's default user agent.
        request = urllib.request.Request(uri, headers={"User-Agent": "curl/8.5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(max_bytes + 1)
    except Exception:
        raise RuntimeError("PRIVATE_INPUT_DOWNLOAD_FAILED:" + label) from None
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("PRIVATE_INPUT_BYTES_DRIFT:" + label)
    return data


def resolve_payload(payload, *, run_id, code_sha, downloader=_download_pinned):
    """Resolve either the legacy inline payload or one pinned manifest locator."""
    if not isinstance(payload, dict):
        raise RuntimeError("PRIVATE_INPUT_PAYLOAD_INVALID")
    _require_identity(payload, run_id, code_sha, "PRIVATE_INPUT_PAYLOAD_IDENTITY_DRIFT")
    manifest = payload.get("manifest")
    if manifest is None:
        return payload
    if "transfers" in payload or "bundle" in payload or "config" in payload:
        raise RuntimeError("PRIVATE_INPUT_MANIFEST_PAYLOAD_AMBIGUOUS")
    raw = downloader(manifest, "manifest", MAX_MANIFEST_BYTES)
    try:
        resolved = json.loads(raw)
    except Exception:
        raise RuntimeError("PRIVATE_INPUT_MANIFEST_JSON_INVALID") from None
    if not isinstance(resolved, dict) or "manifest" in resolved:
        raise RuntimeError("PRIVATE_INPUT_MANIFEST_SHAPE_INVALID")
    _require_identity(resolved, run_id, code_sha, "PRIVATE_INPUT_MANIFEST_IDENTITY_DRIFT")
    return resolved


def _validate_bundle_file(row, names):
    if not isinstance(row, dict):
        raise RuntimeError("PRIVATE_INPUT_BUNDLE_FILE_INVALID")
    name = row.get("name")
    size = row.get("size_bytes")
    digest = row.get("sha256")
    if (not isinstance(name, str) or not SAFE_INPUT_NAME.fullmatch(name) or name in names
            or not isinstance(size, int) or size < 0 or size > MAX_TRANSFER_BYTES
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise RuntimeError("PRIVATE_INPUT_BUNDLE_FILE_INVALID")
    names.add(name)
    return name, size, digest


def _extract_bundle(ref, out, *, downloader=_download_pinned):
    """Download one pinned ZIP and verify every expected inner byte sequence."""
    if not isinstance(ref, dict):
        raise RuntimeError("PRIVATE_INPUT_BUNDLE_INVALID")
    rows = ref.get("files")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_INPUT_FILES:
        raise RuntimeError("PRIVATE_INPUT_BUNDLE_FILE_LIST_INVALID")
    names = set()
    expected = {}
    total_expected = 0
    for row in rows:
        name, size, digest = _validate_bundle_file(row, names)
        expected[name] = (size, digest)
        total_expected += size
    if total_expected > MAX_BUNDLE_UNCOMPRESSED_BYTES:
        raise RuntimeError("PRIVATE_INPUT_BUNDLE_UNCOMPRESSED_LIMIT")

    raw = downloader(ref, "bundle", MAX_BUNDLE_BYTES)
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw), "r")
    except (zipfile.BadZipFile, OSError):
        raise RuntimeError("PRIVATE_INPUT_BUNDLE_ZIP_INVALID") from None
    with archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_INPUT_FILES:
            raise RuntimeError("PRIVATE_INPUT_BUNDLE_ZIP_ENTRY_COUNT_INVALID")
        actual = {}
        total_actual = 0
        for info in infos:
            if (info.is_dir() or not SAFE_INPUT_NAME.fullmatch(info.filename)
                    or info.filename in actual or (info.flag_bits & 0x1)):
                raise RuntimeError("PRIVATE_INPUT_BUNDLE_ZIP_ENTRY_INVALID")
            total_actual += int(info.file_size)
            if total_actual > MAX_BUNDLE_UNCOMPRESSED_BYTES:
                raise RuntimeError("PRIVATE_INPUT_BUNDLE_UNCOMPRESSED_LIMIT")
            actual[info.filename] = info
        if set(actual) != set(expected):
            raise RuntimeError("PRIVATE_INPUT_BUNDLE_ZIP_FILE_SET_DRIFT")
        out.mkdir(parents=True, exist_ok=True)
        verified = []
        for name in sorted(expected):
            size, digest = expected[name]
            info = actual[name]
            if info.file_size != size:
                raise RuntimeError("PRIVATE_INPUT_BUNDLE_INNER_SIZE_DRIFT:" + name)
            with archive.open(info, "r") as source:
                data = source.read(size + 1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError("PRIVATE_INPUT_BUNDLE_INNER_BYTES_DRIFT:" + name)
            (out/name).write_bytes(data)
            verified.append((name, digest))
    return verified


def _decode_contents_api_envelope(body):
    """Decode one GitHub Contents API file response into the encrypted envelope."""
    try:
        metadata = json.loads(body)
    except Exception:
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_API_JSON_INVALID") from None
    if (not isinstance(metadata, dict) or metadata.get("type") != "file"
            or metadata.get("encoding") != "base64"
            or not isinstance(metadata.get("content"), str)):
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_API_SHAPE_INVALID")
    encoded = "".join(metadata["content"].split())
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception:
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_API_CONTENT_INVALID") from None
    if len(raw) > MAX_ENVELOPE_BYTES:
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_TOO_LARGE")
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except Exception:
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_JSON_INVALID") from None
    if not isinstance(envelope, dict):
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_SHAPE_INVALID")
    return envelope


def _envelope_is_current(envelope, public):
    _require_identity(envelope, public["run_id"], public["code_sha"], "PRIVATE_INPUT_ENVELOPE_IDENTITY_DRIFT")
    # A re-run retains its run ID and code SHA but creates a new private key.
    # Wait for the sender's new envelope instead of trying to decrypt stale data.
    return envelope.get("public_key_sha256") == hashlib.sha256(public["public_key_pem"].encode()).hexdigest()


def receive(directory, out, *, repo, branch, run_id, code_sha, timeout):
    public = json.loads((directory / "public.json").read_text())
    _require_identity(public, run_id, code_sha, "PRIVATE_INPUT_PUBLIC_KEY_IDENTITY_DRIFT")
    url = (f"https://api.github.com/repos/{repo}/contents/.github/research_transport/{run_id}.json?ref="
           + urllib.parse.quote(branch, safe=""))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        req = urllib.request.Request(url, headers={
            "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                body = response.read(MAX_CONTENTS_API_BYTES + 1)
            if len(body) > MAX_CONTENTS_API_BYTES:
                raise RuntimeError("PRIVATE_INPUT_ENVELOPE_API_RESPONSE_TOO_LARGE")
            envelope = _decode_contents_api_envelope(body)
            if _envelope_is_current(envelope, public):
                break
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise RuntimeError("PRIVATE_INPUT_ENVELOPE_FETCH_FAILED") from None
        time.sleep(5)
    else:
        raise RuntimeError("PRIVATE_INPUT_HANDOFF_TIMEOUT")
    if envelope.get("run_id") != run_id or envelope.get("code_sha") != code_sha:
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_IDENTITY_DRIFT")
    chunks = envelope.get("chunks")
    plaintext_sha256 = envelope.get("plaintext_sha256")
    if (not isinstance(chunks, list) or len(chunks) > 2048
            or not isinstance(plaintext_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", plaintext_sha256)):
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_SHAPE_INVALID")
    try:
        raw = b"".join(rsa(base64.b64decode(c, validate=True), directory/"private.pem", encrypt=False)
                       for c in chunks if isinstance(c, str))
    except Exception:
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_DECRYPT_FAILED") from None
    if len(chunks) != sum(isinstance(c, str) for c in chunks):
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_SHAPE_INVALID")
    if hashlib.sha256(raw).hexdigest() != plaintext_sha256:
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_INTEGRITY_DRIFT")
    try:
        decrypted = json.loads(raw)
    except Exception:
        raise RuntimeError("PRIVATE_INPUT_PAYLOAD_JSON_INVALID") from None
    payload = resolve_payload(decrypted, run_id=run_id, code_sha=code_sha)
    transfers = payload.get("transfers")
    bundle = payload.get("bundle")
    config = payload.get("config")
    if ((transfers is None) == (bundle is None) or not isinstance(config, dict)):
        raise RuntimeError("PRIVATE_INPUT_PAYLOAD_SHAPE_INVALID")
    out.mkdir(parents=True, exist_ok=True)
    if bundle is not None:
        verified = _extract_bundle(bundle, out)
        for name, digest in verified:
            print("EXACT_PRIVATE_INPUT " + name + " " + digest, flush=True)
    else:
        if not isinstance(transfers, list) or len(transfers) > MAX_INPUT_FILES:
            raise RuntimeError("PRIVATE_INPUT_PAYLOAD_SHAPE_INVALID")
        names = set()
        for transfer in transfers:
            if not isinstance(transfer, dict):
                raise RuntimeError("PRIVATE_INPUT_TRANSFER_INVALID")
            name = transfer.get("name")
            if not isinstance(name, str) or not SAFE_INPUT_NAME.fullmatch(name) or name in names:
                raise RuntimeError("PRIVATE_INPUT_NAME_INVALID")
            names.add(name)
            data = _download_pinned(transfer, name, MAX_TRANSFER_BYTES)
            (out/name).write_bytes(data)
            print("EXACT_PRIVATE_INPUT " + name + " " + transfer["sha256"], flush=True)
    (out/"presentation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (directory/"private.pem").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--code-sha", required=True)
    p = sub.add_parser("encrypt")
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("receive")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--code-sha", required=True)
    p.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.operation == "generate":
        generate(args.directory, args.run_id, args.code_sha)
    elif args.operation == "encrypt":
        encrypt(args.public, args.payload, args.out)
    else:
        receive(args.directory, args.out, repo=args.repo, branch=args.branch,
                run_id=args.run_id, code_sha=args.code_sha, timeout=args.timeout)


if __name__ == "__main__":
    main()
