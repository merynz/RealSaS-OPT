"""One-run RSA-OAEP transport for private download locators on a public CI repo.

The runner retains its private key locally and publishes only the public key.
Ciphertext is bound to the run ID and exact code SHA. Locators are never logged
or committed as plaintext; scientific bytes still require their pinned SHA256.
"""
import argparse
import base64
import hashlib
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
                              "plaintext_sha256": hashlib.sha256(raw).hexdigest(), "chunks": chunks}) + "\n")


def receive(directory, out, *, repo, branch, run_id, code_sha, timeout):
    url = (f"https://api.github.com/repos/{repo}/contents/.github/research_transport/{run_id}.json?ref="
           + urllib.parse.quote(branch, safe=""))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
            "Accept": "application/vnd.github.raw+json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                envelope = json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise RuntimeError("PRIVATE_INPUT_ENVELOPE_FETCH_FAILED") from None
        time.sleep(5)
    else:
        raise RuntimeError("PRIVATE_INPUT_HANDOFF_TIMEOUT")
    if envelope["run_id"] != run_id or envelope["code_sha"] != code_sha:
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_IDENTITY_DRIFT")
    if len(envelope["chunks"]) > 2048:
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_TOO_LARGE")
    raw = b"".join(rsa(base64.b64decode(c, validate=True), directory/"private.pem", encrypt=False)
                   for c in envelope["chunks"])
    if hashlib.sha256(raw).hexdigest() != envelope["plaintext_sha256"]:
        raise RuntimeError("PRIVATE_INPUT_ENVELOPE_INTEGRITY_DRIFT")
    payload = json.loads(raw)
    if payload["run_id"] != run_id or payload["code_sha"] != code_sha:
        raise RuntimeError("PRIVATE_INPUT_PAYLOAD_IDENTITY_DRIFT")
    out.mkdir(parents=True, exist_ok=True)
    names = set()
    for transfer in payload["transfers"]:
        name, uri = transfer["name"], transfer["download_url"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in names:
            raise RuntimeError("PRIVATE_INPUT_NAME_INVALID")
        names.add(name)
        parsed = urllib.parse.urlsplit(uri)
        if parsed.scheme != "https" or not (parsed.hostname or "").endswith((".blob.core.windows.net", ".amazonaws.com", ".oaiusercontent.com")):
            raise RuntimeError("PRIVATE_INPUT_DOWNLOAD_HOST_INVALID")
        # GitHub log masking is defense in depth; never print locators.
        print("::add-mask::" + uri, flush=True)
        try:
            # The signed-download gateway rejects urllib's default user agent.
            download = urllib.request.Request(uri, headers={"User-Agent": "curl/8.5.0"})
            with urllib.request.urlopen(download, timeout=60) as response:
                data = response.read(64*1024*1024 + 1)
        except Exception:
            raise RuntimeError("PRIVATE_INPUT_DOWNLOAD_FAILED:" + name) from None
        if len(data) != transfer["size_bytes"] or hashlib.sha256(data).hexdigest() != transfer["sha256"]:
            raise RuntimeError("PRIVATE_INPUT_BYTES_DRIFT:" + name)
        (out/name).write_bytes(data)
        print("EXACT_PRIVATE_INPUT " + name + " " + transfer["sha256"], flush=True)
    (out/"presentation_config.json").write_text(json.dumps(payload["config"], indent=2) + "\n")
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
