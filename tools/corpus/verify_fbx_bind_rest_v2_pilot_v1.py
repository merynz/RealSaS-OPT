"""Independent original-import versus V2 raw FBX bind-rest proof candidate.

The imported Blender scene is a fixed historical reference, not training truth
minted here. Exactly index-matched per-object vertices and original byte SHA are
required. This witness does NOT replace the eight-view observation court.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _blocks(vertices, objects, *, prefix):
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1:] != (3,) or not np.isfinite(verts).all():
        raise ValueError(prefix + "_VERTEX_LAYOUT")
    offset = 0
    blocks = {}
    for row in objects:
        count, name = row["vertex_count"], row["name"]
        if type(count) is not int or count < 0 or name in blocks:
            raise ValueError(prefix + "_OBJECT_IDENTITY")
        if offset + count > len(verts):
            raise ValueError(prefix + "_VERTEX_COVERAGE")
        blocks[name] = verts[offset:offset+count]
        offset += count
    if offset != len(verts):
        raise ValueError(prefix + "_VERTEX_COVERAGE")
    return blocks


def validate(raw_source, imported_path, v2_path, *, source_sha, imported_sha,
             max_relative_vertex_error):
    raw_source, imported_path, v2_path = map(Path, (raw_source, imported_path, v2_path))
    if sha256(raw_source) != source_sha or sha256(imported_path) != imported_sha:
        raise ValueError("FBX_V2_PILOT_PARENT_BYTES_DRIFT")
    imported = json.loads(imported_path.read_text())
    candidate = json.loads(v2_path.read_text())
    if (imported.get("import", {}).get("source_sha256") != source_sha
            or imported.get("import", {}).get("pose_policy") != "SOURCE_BIND_REST__NO_ANIMATION"
            or candidate.get("reader_schema") != "RealSaS.SourceFBXBindRestReader.v2"
            or candidate.get("bone_count") != imported.get("bone_count")):
        raise ValueError("FBX_V2_PILOT_BIND_SCOPE")
    expected = _blocks(imported["vertices"], imported["objects"],prefix="FBX_V2_IMPORT")
    observed = _blocks(candidate["vertices"],candidate["objects"],prefix="FBX_V2_SOURCE")
    if expected.keys() != observed.keys():
        raise ValueError("FBX_V2_PILOT_OBJECT_SET_DRIFT")
    scale = max(float(np.linalg.norm(np.ptp(np.asarray(imported["vertices"],np.float64),axis=0))),1e-9)
    errors={}
    for name in sorted(expected):
        a,b=expected[name],observed[name]
        if a.shape != b.shape:
            raise ValueError("FBX_V2_PILOT_OBJECT_VERTEX_DRIFT:"+name)
        errors[name]=float(np.max(np.linalg.norm(a-b,axis=1),initial=0.0)/scale)
    maximum=max(errors.values(),default=0.)
    limit=float(max_relative_vertex_error)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("FBX_V2_PILOT_INVALID_UNCHANGED_LIMIT")
    return {
        "schema":"RealSaS.Corpus370FBXBindRestV2Pilot.v1",
        "status":"NUMERIC_PARITY_SCREEN__NOT_OBSERVATION_ADMISSION",
        "source_sha256":source_sha,
        "original_import_sha256":imported_sha,
        "v2_probe_output_sha256":sha256(v2_path),
        "source_object_count":len(observed),
        "source_bone_count":candidate["bone_count"],
        "bone_parent_attachment_mesh_count":candidate.get("bone_parent_attachment_mesh_count"),
        "max_per_index_vertex_error_over_source_scale":maximum,
        "policy_limit_unchanged":limit,
        "within_original_policy_numeric_gate":maximum<=limit,
        "per_object_error_over_source_scale":errors,
        "training_authorized":False,
        "admission_minted":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--imported",type=Path,required=True)
    p.add_argument("--v2",type=Path,required=True)
    p.add_argument("--source-sha",required=True)
    p.add_argument("--import-sha",required=True)
    p.add_argument("--original-limit",type=float,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=validate(args.source,args.imported,args.v2,source_sha=args.source_sha,
                    imported_sha=args.import_sha,
                    max_relative_vertex_error=args.original_limit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":main()
