from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from typing import Protocol


@dataclass(frozen=True)
class StoredObject:
    content_sha256: str
    storage_key: str
    size_bytes: int


class ArtifactStore(Protocol):
    def put_bytes(self, data: bytes) -> StoredObject: ...
    def get_bytes(self, content_sha256: str) -> bytes: ...
    def verify(self, content_sha256: str) -> bool: ...


class LocalContentAddressedStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _require_hash(value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid sha256")
        return value

    def path_for(self, content_sha256: str) -> Path:
        digest = self._require_hash(content_sha256)
        return self.root / "cas" / "sha256" / digest[:2] / digest[2:4] / digest

    def put_bytes(self, data: bytes) -> StoredObject:
        digest = sha256(data).hexdigest(); target = self.path_for(digest); target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != data: raise RuntimeError("CAS_HASH_COLLISION_OR_CORRUPTION")
        else:
            fd, temp_name = tempfile.mkstemp(prefix=".realsas-cas-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data); handle.flush(); os.fsync(handle.fileno())
                os.replace(temp_name, target)
            finally:
                if os.path.exists(temp_name): os.unlink(temp_name)
        if not self.verify(digest): raise RuntimeError("CAS_POST_WRITE_VERIFICATION_FAILED")
        return StoredObject(digest, target.relative_to(self.root).as_posix(), len(data))

    def get_bytes(self, content_sha256: str) -> bytes:
        data = self.path_for(content_sha256).read_bytes()
        if sha256(data).hexdigest() != content_sha256: raise RuntimeError("CAS_CONTENT_HASH_MISMATCH")
        return data

    def verify(self, content_sha256: str) -> bool:
        path = self.path_for(content_sha256)
        if not path.is_file(): return False
        h = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""): h.update(chunk)
        return h.hexdigest() == content_sha256
