from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_SHA = "81634b7db6dbae3f7cc30d7f6942ace9be841416dd1db833185884d3e10584c5"
PREFIX_PARTS = (
    "part_00.txt",
    "part_01.txt",
    "part_02.txt",
    "part_03a.txt",
    "part_03b.txt",
    "part_04a.txt",
)


def main() -> None:
    chunk_dir = HERE / "fixture_chunks"
    prefix = "".join((chunk_dir / name).read_text(encoding="ascii").strip() for name in PREFIX_PARTS)
    monolith = (HERE / "CONSUMER_INTERLOCK_COMPILER_FIXTURE_V2.b64").read_text(encoding="ascii").strip()

    prefix_state = hashlib.sha256(prefix.encode("ascii"))
    matches: list[int] = []
    for i in range(len(monolith) + 1):
        h = prefix_state.copy()
        h.update(monolith[i:].encode("ascii"))
        if h.hexdigest() == TARGET_SHA:
            matches.append(i)

    report = {
        "schema": "RealSaS.ConsumerInterlock.FixtureTailRecovery.v1",
        "target_transport_sha256": TARGET_SHA,
        "prefix_parts": list(PREFIX_PARTS),
        "prefix_length": len(prefix),
        "prefix_sha256": hashlib.sha256(prefix.encode("ascii")).hexdigest(),
        "corrupt_monolith_length": len(monolith),
        "match_count": len(matches),
        "match_indices": matches,
        "status": "NO_MATCH",
    }

    if len(matches) == 1:
        i = matches[0]
        tail = monolith[i:]
        transport = prefix + tail
        raw = zlib.decompress(base64.b64decode(transport, validate=True))
        fixture = json.loads(raw)
        if fixture.get("schema") != "RealSaS.ConsumerInterlock.CompilerFixture.v2":
            raise RuntimeError("recovered transport decodes but fixture schema is wrong")
        if hashlib.sha256(transport.encode("ascii")).hexdigest() != TARGET_SHA:
            raise RuntimeError("recovered transport SHA changed after validation")
        (HERE / "RECOVERED_PART_04B.txt").write_text(tail, encoding="ascii")
        report.update(
            {
                "status": "UNIQUE_RECOVERY",
                "recovered_tail_length": len(tail),
                "recovered_tail_sha256": hashlib.sha256(tail.encode("ascii")).hexdigest(),
                "recovered_fixture_schema": fixture["schema"],
                "recovered_asset_id": fixture.get("asset_id"),
            }
        )
    elif len(matches) > 1:
        report["status"] = "AMBIGUOUS_RECOVERY"

    out = HERE / "FIXTURE_TAIL_RECOVERY_V1.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    if report["status"] != "UNIQUE_RECOVERY":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
