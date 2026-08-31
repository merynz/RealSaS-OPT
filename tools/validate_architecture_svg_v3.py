from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "canonical" / "REALSAS_END_TO_END_CANONICAL_ARCHITECTURE_20260901.svg"
V3 = ROOT / "canonical" / "SYSTEM_ARCHITECTURE_V3_20260901.md"
ROADMAP = ROOT / "canonical" / "HUMAN_READABLE_ROADMAP_20260901.md"

REQUIRED_IDS = {
    "legend",
    "input",
    "iris",
    "obs",
    "gsa",
    "S",
    "BG",
    "geppetto",
    "Gstar",
    "rigqual",
    "G",
    "BA",
    "arachne",
    "Wstar",
    "skinqual",
    "W",
    "meshprod",
    "Mstar",
    "meshqual",
    "M",
    "binder",
    "B",
    "Y",
    "proof",
    "runtimepkg",
    "repair",
    "native",
    "clean-c0",
    "r6",
    "teacher",
    "codec",
    "histmax",
    "vendor",
    "cdt",
    "bbw",
    "arap",
    "xpbd",
    "heavyproof",
    "cpp",
    "registry",
    "promotion",
    "dead",
    "gates",
    "migration",
}

REQUIRED_TEXT = (
    "ObservationEvidenceIR",
    "RiggingSurfaceIR S",
    "SkeletonProposalIR G*",
    "QualifiedSkeletonIR G",
    "SkinProposalIR W*",
    "QualifiedSkinIR W",
    "MeshDiscretizationCandidateIR M*",
    "QualifiedEditableMeshIR M",
    "QualifiedMeshSkinIR B",
    "CanonicalPuppetGraph.v2 Y",
    "ProofFrame",
    "RuntimePackageIR",
    "CDT",
    "BBW",
    "ARAP",
    "XPBD",
    "EXECUTABLE_SOLVERS = {}",
    "DEAD_SUPERSEDED_CONFIRMED",
    "NO learned optimizer step currently authorized",
    "13/13 intended vendored records reconciled",
)


def main() -> int:
    if not SVG.is_file() or not V3.is_file() or not ROADMAP.is_file():
        raise SystemExit("missing architecture V3 artifact")

    root = ET.parse(SVG).getroot()
    if not root.tag.endswith("svg"):
        raise SystemExit("architecture artifact is not SVG")

    ids = {x.attrib.get("id") for x in root.iter() if x.attrib.get("id")}
    missing_ids = sorted(REQUIRED_IDS - ids)
    if missing_ids:
        raise SystemExit(f"missing required SVG ids: {missing_ids}")

    text = SVG.read_text(encoding="utf-8")
    missing_text = [x for x in REQUIRED_TEXT if x not in text]
    if missing_text:
        raise SystemExit(f"missing required architecture labels: {missing_text}")

    # Explicitly guard the most dangerous authority confusions.
    forbidden = (
        "EXECUTABLE_SOLVERS = {CDT",
        "IRIS owns canonical skeleton",
        "Arachne owns canonical skin",
        "runtime owns canonical product",
        "hidden completion authorized",
    )
    present_forbidden = [x for x in forbidden if x in text]
    if present_forbidden:
        raise SystemExit(f"forbidden authority wording in SVG: {present_forbidden}")

    v3 = V3.read_text(encoding="utf-8")
    roadmap = ROADMAP.read_text(encoding="utf-8")
    # These are V3's own normative human-readable architecture classes/terms.
    # Machine-status tokens from subordinate migration artifacts are validated in
    # their own closures; V3 intentionally defines its visual vocabulary explicitly.
    for required in (
        "CanonicalPuppetGraph.v2",
        "HISTORICAL EXTERNAL BYTE / NUMERICAL AUTHORITY",
        "MWB-1",
        "R6",
        "No learned optimizer step is authorized",
    ):
        if required not in v3:
            raise SystemExit(f"V3 authority missing: {required}")

    if "REALSAS_END_TO_END_CANONICAL_ARCHITECTURE_20260901.svg" not in roadmap:
        raise SystemExit("roadmap does not point to V3 SVG")

    sha = hashlib.sha256(SVG.read_bytes()).hexdigest()
    print("SVG_XML_PARSE=PASS")
    print(f"REQUIRED_NODE_IDS={len(REQUIRED_IDS)}/{len(REQUIRED_IDS)} PASS")
    print(f"AUTHORITY_LABELS={len(REQUIRED_TEXT)}/{len(REQUIRED_TEXT)} PASS")
    print(f"SVG_SHA256={sha}")
    print("ARCHITECTURE_V3_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
