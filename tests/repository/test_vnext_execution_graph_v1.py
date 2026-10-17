from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def _graph():return json.loads((ROOT/"canonical"/"VNEXT_EXECUTION_GRAPH_V1.json").read_text())
def _ancestors(by,node):
    out=set();stack=list(by[node]["depends_on"])
    while stack:
        x=stack.pop()
        if x in out:continue
        out.add(x);stack.extend(by[x]["depends_on"])
    return out

def test_vnext_graph_has_one_and_only_one_canonical_mechanics_authority():
    g=_graph();rows=g["stages"];canonical=[x for x in rows if x["authority"]=="SOLE_CANONICAL_M_G_W"];assert len(canonical)==1;assert canonical[0]["id"]=="10_CANONICAL_MECHANICS_SELECTED";assert all("CAA" not in x["id"] for x in rows)
def test_every_graphics_runtime_authority_is_downstream_of_single_canonical_mechanics():
    g=_graph();by={x["id"]:x for x in g["stages"]}
    for row in g["stages"]:
        if row["id"] in {"04_GRAPHICS_RELATION_EVIDENCE_SEALED"}:continue
        if row["group"] in {"graphics","runtime","closure"} and row["id"]!="01_SOURCE_OBSERVATION_SEALED":
            if row["authority"] in {"LEARNED_EVIDENCE_ONLY"}:continue
            assert "10_CANONICAL_MECHANICS_SELECTED" in _ancestors(by,row["id"]),row["id"]
def test_candidate_dynamic_probe_precedes_canonical_and_production_proof_follows_it():
    g=_graph();by={x["id"]:x for x in g["stages"]};assert "09_CANDIDATE_MECHANICAL_PROBES" in _ancestors(by,"10_CANONICAL_MECHANICS_SELECTED");assert "10_CANONICAL_MECHANICS_SELECTED" in _ancestors(by,"15_PRODUCTION_DYNAMIC_PROOF");assert by["09_CANDIDATE_MECHANICAL_PROBES"]["production_motion_proof"] is False;assert by["15_PRODUCTION_DYNAMIC_PROOF"]["candidate_selection_probe"] is False
def test_runtime_requires_graphics_content_not_hash_only_binding():
    g=_graph();by={x["id"]:x for x in g["stages"]};assert by["18_RSS_GRAPHICS_MATERIALIZED"]["graphics_content_required"] is True;assert by["19_NATIVE_GRAPHICS_PLAYBACK_PROVEN"]["hash_only_binding_forbidden"] is True
