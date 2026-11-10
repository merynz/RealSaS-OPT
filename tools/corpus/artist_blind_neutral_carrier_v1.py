"""Leakage-free *topology* control: one neutral grid, NOT an art reconstruction.

Starting from a human-art-origin merged triangle sample, retain only its global
2D bounds, and construct a regular single-component carrier independent of
original artist attachment/slot/face incidence. This deliberately drops all
original appearance and authored face addresses. A geometry-only model cannot
recover the original artist's 2D layer partition from it. Gold comparison
MUST ABSTAIN until a raster-layer address adapter and reference compositor
exist; never project old author face IDs onto this grid without a new court.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from tools.corpus.author_spine_blind_flatten_v1 import (
    MERGED,canonical_bytes,digest
)
from tools.corpus.artist_blind_topology_leakage_audit_v1 import (
    predict
)

SCHEMA="RealSaS.AnonymousNeutralContinuousCanvasCarrier.v1"

def require(ok,reason):
    if not ok:raise ValueError(reason)

def build(anonymous_merged,*,cells=24):
    require(anonymous_merged.get("schema")==MERGED
            and anonymous_merged.get("original_art_layer_count_visible") is False
            and anonymous_merged.get("original_slot_attachment_bone_names_present") is False
            and anonymous_merged.get("source_composite_parity_qualified") is False
            and anonymous_merged.get("training_authorized") is False
            and type(cells) is int and 2<=cells<=128,
            "ARTIST_NEUTRAL_CARRIER_NOT_FROM_SEALED_ANONYMOUS_SOURCE")
    xyz=np.asarray(anonymous_merged["vertices_setup_xy"],np.float64)
    require(xyz.ndim==2 and xyz.shape[1]==2 and len(xyz)>3
            and np.isfinite(xyz).all(),"ARTIST_NEUTRAL_CARRIER_INVALID_SOURCE_BOUNDS")
    lo=xyz.min(axis=0);hi=xyz.max(axis=0)
    require(np.all(hi-lo>1e-8),"ARTIST_NEUTRAL_CARRIER_ZERO_AREA_SOURCE")
    xs=np.linspace(lo[0],hi[0],cells+1)
    ys=np.linspace(lo[1],hi[1],cells+1)
    vertices=np.array([[x,y] for y in ys for x in xs],dtype=np.float64)
    faces=[]
    for y in range(cells):
        for x in range(cells):
            i=y*(cells+1)+x
            faces.append([i,i+1,i+cells+1])
            faces.append([i+1,i+cells+2,i+cells+1])
    out={
        "schema":SCHEMA,
        "input_kind":"NEUTRAL_2D_GRID_CARRIER_NO_ORIGINAL_AUTHOR_TOPOLOGY_OR_ART_PIXEL_CAA",
        "source_anonymous_merged_surface_SHA256":digest(canonical_bytes(anonymous_merged)),
        "source_bounds_setup_xy":list(zip(lo.tolist(),hi.tolist())),
        "vertices_setup_xy":vertices.tolist(),
        "faces":faces,
        "grid_resolution_cells_per_axis":cells,
        "original_author_face_incidence_preserved":False,
        "original_author_part_count_visible":False,
        "original_source_slot_name_or_bone_id_present":False,
        "original_atlas_or_raster_appearance_available":False,
        "original_source_face_to_grid_face_correspondence":False,
        "artist_2d_masks_projected_onto_grid":False,
        "source_native_rgba_alpha_parity":False,
        "genuine_author_drawable_recovery_measured":False,
        "training_authorized":False
    }
    return out

def audit_connected(out):
    require(out["schema"]==SCHEMA
            and out["original_author_face_incidence_preserved"] is False
            and out["original_source_face_to_grid_face_correspondence"] is False
            and out["artist_2d_masks_projected_onto_grid"] is False,
            "ARTIST_NEUTRAL_CARRIER_GOLD_ADDRESS_MINT_ATTEMPT")
    # Use the same generic graph connectivity algorithm as our leaked-source
    # baseline, but on topology-independent carrier vertices/faces.
    carrier={
        "schema":MERGED,
        "source_composite_parity_qualified":False,
        "original_art_layer_count_visible":False,
        "original_slot_attachment_bone_names_present":False,
        "training_authorized":False,
        "vertices_setup_xy":out["vertices_setup_xy"],
        "faces":out["faces"]
    }
    p=predict(carrier)
    require(p["predicted_topological_component_count"]==1,
            "ARTIST_NEUTRAL_CARRIER_NOT_ONE_CONNECTED_COMPONENT")
    return {
        "schema":"RealSaS.AuthorTopologyLeakageNeutralCarrierControl.v1",
        "status":"CONTINUOUS_CARRIER_CONNECTEDNESS_ONE__NO_APPEARANCE_OR_ARTIST_LAYER_EVIDENCE",
        "anonymous_continuous_carrier_SHA256":digest(canonical_bytes(out)),
        "one_connected_component":True,
        "original_artist_part_boundaries_by_topology_removed":True,
        "native_source_rgba_reconstruction_qualified":False,
        "original_author_layer_pixel_masks_available":False,
        "source_artist_face_labels_comparable_to_grid_faces":False,
        "artist_drawable_recovery_accuracy":None,
        "generic_geometry_only_ownership_proven":False,
        "training_authorized":False
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--anonymous-input",type=Path,required=True)
    p.add_argument("--neutral-carrier",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--cells",type=int,default=24)
    args=p.parse_args()
    m=json.loads(args.anonymous_input.read_text())
    out=build(m,cells=args.cells)
    result=audit_connected(out)
    for path,artifact in ((args.neutral_carrier,out),(args.receipt,result)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(canonical_bytes(artifact))
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":
    main()
