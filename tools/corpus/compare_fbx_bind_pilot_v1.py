"""Bounded, independent numerical comparison of raw-FBX bind candidates.

No reader mode is selected and no family/admission/training authority is issued.
Matching uses original per-object names and bidirectional nearest-vertex distances.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hausdorff(a, b, scale):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1:] != (3,) or
            b.shape[1:] != (3,) or not len(a) or not len(b) or
            not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("FBX_PILOT_VERTEX_LAYOUT")
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()) / scale)


def compare(source, imported, diagnostic, source_sha, import_sha):
    source, imported, diagnostic = map(Path, (source, imported, diagnostic))
    if sha256(source) != source_sha or sha256(imported) != import_sha:
        raise ValueError("FBX_PILOT_PARENT_DRIFT")
    native = json.loads(imported.read_text())
    diag = json.loads(diagnostic.read_text())
    if (diag.get("schema") != "RealSaS.FBXBindPositionDiagnostic.v1"
            or diag.get("authority", {}).get("training_authorized") is not False
            or diag.get("authority", {}).get("candidate_bind_policy_selected") is not False
            or diag.get("bone_count") != native.get("bone_count")):
        raise ValueError("FBX_PILOT_DIAGNOSTIC_SCOPE")
    if (native.get("import", {}).get("source_sha256") != source_sha
            or native.get("import", {}).get("pose_policy") != "SOURCE_BIND_REST__NO_ANIMATION"):
        raise ValueError("FBX_PILOT_IMPORT_SCOPE")
    blocks, offset = {}, 0
    vertices = np.asarray(native["vertices"], np.float64)
    for obj in native["objects"]:
        name, count = obj["name"], int(obj["vertex_count"])
        if name in blocks or count < 1 or offset+count > len(vertices):
            raise ValueError("FBX_PILOT_DUPLICATE_OR_INVALID_IMPORT_OBJECT")
        blocks[name] = vertices[offset:offset+count]
        offset += count
    if offset != len(vertices):
        raise ValueError("FBX_PILOT_IMPORT_VERTEX_COVERAGE")
    rows = []
    names = set()
    for raw in diag["objects"]:
        name = raw["name"]
        if name in names:
            raise ValueError("FBX_PILOT_DUPLICATE_SOURCE_OBJECT")
        names.add(name)
        entry = {"name": name, "matched_import_object": name in blocks,
                 "vertex_count": raw["vertex_count"],
                 "skin_deformer_count": raw["skin_deformer_count"],
                 "cluster_bind_position_count": raw["cluster_bind_position_count"]}
        if name in blocks:
            default = np.asarray(raw["default_world"], np.float64)
            expected = blocks[name]
            if len(default) != int(raw["vertex_count"]) or len(default) != len(expected):
                entry["reason"] = "OBJECT_VERTEX_COUNT_DRIFT"
            else:
                scale = max(float(np.linalg.norm(np.ptp(expected, axis=0))), 1e-9)
                entry["default_relative_error"] = hausdorff(default, expected, scale)
                parent_rows = raw.get("bone_parent_bind_world") or []
                entry["source_parent_node"] = raw.get("source_parent_node")
                entry["attached_bone"] = raw.get("attached_bone")
                entry["bone_parent_bind_relative_error"] = (
                    hausdorff(parent_rows, expected, scale)
                    if len(parent_rows) == len(expected)
                       and all(v is not None for v in parent_rows)
                    else None
                )
                bind_rows = raw["cluster_bind_world"]
                if len(bind_rows) == len(expected) and all(v is not None for v in bind_rows):
                    entry["cluster_bind_relative_error"] = hausdorff(bind_rows, expected, scale)
                else:
                    entry["cluster_bind_relative_error"] = None
                    entry["reason"] = "INCOMPLETE_CLUSTER_BIND_COVERAGE"
        rows.append(entry)
    return {"schema": "RealSaS.Corpus370FBXBindComparison.v1",
            "status": "BOUNDED_SOURCE_BIND_DIAGNOSTIC__NO_ADMISSION",
            "source_sha256": source_sha, "original_import_sha256": import_sha,
            "source_object_names_match": names == set(blocks),
            "per_object": rows,
            "authority": {"source_data_mutated": False, "training_authorized": False,
                          "new_family_authority_minted": False,
                          "fbx_bind_policy_selected": False}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--imported",type=Path,required=True)
    p.add_argument("--diagnostic",type=Path,required=True)
    p.add_argument("--source-sha256",required=True)
    p.add_argument("--import-sha256",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=compare(a.source,a.imported,a.diagnostic,a.source_sha256,a.import_sha256)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__": main()
