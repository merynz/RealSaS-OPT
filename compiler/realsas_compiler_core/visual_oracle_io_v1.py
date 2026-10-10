"""Content-addressed research output I/O; carries no acceptance decision."""
import hashlib
import json
from pathlib import Path

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_ref(path: Path, schema: str) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path.read_bytes()),
            "schema": schema, "authority_class": "RESEARCH_VISUAL_ORACLE"}


def write_json(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return file_ref(path, value["schema"])


