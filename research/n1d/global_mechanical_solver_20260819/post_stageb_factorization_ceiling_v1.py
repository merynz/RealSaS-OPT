from __future__ import annotations

import base64
import hashlib
from pathlib import Path

EXPECTED_DECODED_SHA256 = "caffa717db433a56109f16eefa5ce68c4e13f24722ac40c46b85b2f6d00473a7"


def main() -> None:
    here = Path(__file__).resolve().parent
    parts = sorted((here / "source_bundle_b64").glob("post_stageb_factorization_ceiling_v1.py.b64.part*"))
    if not parts:
        raise RuntimeError("factorization ceiling source bundle parts are missing")
    encoded = "".join(part.read_text(encoding="ascii").strip() for part in parts)
    source = base64.b64decode(encoded, validate=True)
    actual = hashlib.sha256(source).hexdigest()
    if actual != EXPECTED_DECODED_SHA256:
        raise RuntimeError(f"decoded source SHA mismatch: {actual} != {EXPECTED_DECODED_SHA256}")
    code = compile(source, "post_stageb_factorization_ceiling_v1.decoded.py", "exec")
    globals_dict = {"__name__": "__main__", "__file__": str(here / "post_stageb_factorization_ceiling_v1.decoded.py")}
    exec(code, globals_dict, globals_dict)


if __name__ == "__main__":
    main()
