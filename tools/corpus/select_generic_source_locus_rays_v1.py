#!/usr/bin/env python3
"""Geometry-only, deterministic view-diverse sampling for source-locus pilots.

No RGB, source domain names, materials, semantics, rig IDs, hidden oracle
outcomes or training labels enter selection. The source triangle indices are
privileged sampling addresses, never student inputs. Requires cross-view
coverage so two adjacent pixels in view zero cannot masquerade as two
independent ownership examples.
"""
from __future__ import annotations

VIEW_ORDER=(0,4,2,6,1,5,3,7)
VALID="VALID_GEOMETRIC_FULL_STACK_CANDIDATE"

def require(ok,code):
    if not ok: raise ValueError(code)

def select_rays(rows,*,maximum=2):
    require(type(maximum) is int and 2<=maximum<=8,
            "GENERIC_PILOT_VIEW_QUOTA")
    require(isinstance(rows,list) and bool(rows),
            "GENERIC_PILOT_RAYS_REQUIRED")
    by_view={i:[] for i in range(8)}
    for r in rows:
        require(isinstance(r,dict),"GENERIC_PILOT_RAY_OBJECT")
        if r.get("status")!=VALID or len(r.get("hits") or [])<2:
            continue
        require(r.get("complete") is True and r.get("has_depth_tie") is False,
                "GENERIC_PILOT_FALSE_VALID_GEOMETRIC_RAY")
        v=r.get("view_index")
        require(type(v) is int and 0<=v<8,"GENERIC_PILOT_VIEW_INDEX")
        require(type(r.get("pixel_x")) is int and r["pixel_x"]>=0
                and type(r.get("pixel_y")) is int and r["pixel_y"]>=0,
                "GENERIC_PILOT_PIXEL")
        f=[hit.get("face_index_teacher_only") for hit in r["hits"][:2]]
        require(all(type(x) is int and x>=0 for x in f),
                "GENERIC_PILOT_SOURCE_FACES")
        # A self-face is not a two-source causal comparison.
        if f[0]!=f[1]:
            by_view[v].append(r)
    present=[v for v in VIEW_ORDER if by_view[v]]
    require(len(present)>=2,"GENERIC_PILOT_TWO_INDEPENDENT_VIEWS_REQUIRED")
    chosen=[];seen_pairs=set()
    # Each view can contribute at most one ray. Rotation coverage is
    # mandatory before any second sample from an already-used viewpoint.
    for v in present[:maximum]:
        options=by_view[v]
        def sort_key(row):
            a,b=[h["face_index_teacher_only"] for h in row["hits"][:2]]
            pair=(min(a,b),max(a,b))
            return (
                pair in seen_pairs,
                abs(a-b)==1,
                min(a,b),max(a,b),
                row["pixel_x"],row["pixel_y"],
            )
        best=min(options,key=sort_key)
        a,b=[h["face_index_teacher_only"] for h in best["hits"][:2]]
        seen_pairs.add((min(a,b),max(a,b)))
        chosen.append(best)
    require(len({r["view_index"] for r in chosen})==len(chosen)>=2,
            "GENERIC_PILOT_VIEW_DIVERSITY_LOST")
    return chosen
