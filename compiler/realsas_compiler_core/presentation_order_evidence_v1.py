from __future__ import annotations

"""Typed, fail-closed presentation-order evidence on semantic M owners.

Source overlap establishes the intended directed relation for one source view.
A relation becomes usable research evidence only when recorded motion exhibits
an actual reversal of that same directed owner pair in that same view.

This module does not modify geometry, topology, motion, CAA, or runtime state.
"""

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

SCHEMA = "RealSaS.PresentationOrderEvidence.v1"


@dataclass(frozen=True)
class PresentationOrderEvidenceV1:
    front_owner_id: str
    back_owner_id: str
    view_index: int
    valid_frame_indices: tuple[int, ...]
    confidence: float
    source_support_count: int
    observed_reversed_pixel_count: int
    peak_reversal_fraction: float
    schema_version: str = SCHEMA

    def to_dict(self):
        row = asdict(self)
        row["valid_frame_indices"] = list(self.valid_frame_indices)
        return row


def _label(owner_id: str) -> str:
    parts = str(owner_id).split(":")
    return parts[1] if len(parts) >= 3 and parts[0] == "OWNER" else ""


def _has_cycle(edges: Iterable[tuple[str, str]]) -> bool:
    adjacency: dict[str, set[str]] = {}
    indegree: dict[str, int] = {}
    for a, b in edges:
        if a == b:
            return True
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set())
    for node in adjacency:
        indegree[node] = 0
    for a, children in adjacency.items():
        for b in children:
            indegree[b] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    seen = 0
    while ready:
        node = ready.pop()
        seen += 1
        for child in adjacency[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return seen != len(indegree)


def compile_presentation_order_evidence_v1(
    *,
    source_relations: Iterable[Mapping],
    pose_conflicts: Iterable[Mapping],
    minimum_source_confidence: float = 0.58,
    minimum_source_support: int = 8,
    minimum_total_reversed_pixels: int = 8,
    minimum_peak_reversal_fraction: float = 0.80,
):
    """Compile source order + observed motion reversals into typed evidence.

    Qualification is exact-owner, exact-view and directed.  Reverse-direction
    contradictions are rejected.  Any frame whose active directed graph forms a
    cycle is removed from all affected edges; an edge with no surviving frame
    domain is dropped.
    """
    conflicts_by_key: dict[tuple[str, str, int], list[Mapping]] = {}
    for row in pose_conflicts:
        if row.get("relation_kind") != "POSE_ORDER_CONFLICT":
            continue
        key = (
            str(row.get("a_owner_id", "")),
            str(row.get("b_owner_id", "")),
            int(row.get("view_index", -1)),
        )
        conflicts_by_key.setdefault(key, []).append(row)

    provisional: dict[tuple[str, str, int], dict] = {}
    rejected = {
        "not_source_front": 0,
        "same_label": 0,
        "low_source_confidence": 0,
        "low_source_support": 0,
        "no_matching_reversal": 0,
        "low_reversed_pixels": 0,
        "low_peak_reversal_fraction": 0,
        "reverse_direction_contradiction": 0,
        "cycle_frame": 0,
        "empty_frame_domain": 0,
    }

    for src in source_relations:
        if src.get("relation_kind") != "FRONT_OF_SOURCE_EVIDENCE":
            rejected["not_source_front"] += 1
            continue
        a = str(src.get("a_owner_id", ""))
        b = str(src.get("b_owner_id", ""))
        view = int(src.get("view_index", -1))
        if not a or not b or a == b or not (0 <= view < 8):
            rejected["same_label"] += 1
            continue
        if _label(a) == _label(b):
            rejected["same_label"] += 1
            continue
        source_conf = float(src.get("confidence", 0.0))
        source_support = int(src.get("support_count", 0))
        if source_conf < minimum_source_confidence:
            rejected["low_source_confidence"] += 1
            continue
        if source_support < minimum_source_support:
            rejected["low_source_support"] += 1
            continue
        rows = conflicts_by_key.get((a, b, view), [])
        if not rows:
            rejected["no_matching_reversal"] += 1
            continue
        reversed_total = 0
        peak = 0.0
        frames: set[int] = set()
        for row in rows:
            meta = row.get("metadata") or {}
            reversed_pixels = int(meta.get("reversed_pixels", row.get("support_count", 0)))
            overlap_pixels = int(meta.get("overlap_pixels", 0))
            if reversed_pixels <= 0:
                continue
            reversed_total += reversed_pixels
            if overlap_pixels > 0:
                peak = max(peak, reversed_pixels / overlap_pixels)
            if "frame_index" in row:
                frames.add(int(row["frame_index"]))
        if reversed_total < minimum_total_reversed_pixels:
            rejected["low_reversed_pixels"] += 1
            continue
        if peak < minimum_peak_reversal_fraction:
            rejected["low_peak_reversal_fraction"] += 1
            continue
        if not frames:
            rejected["no_matching_reversal"] += 1
            continue
        provisional[(a, b, view)] = {
            "front": a,
            "back": b,
            "view": view,
            "frames": frames,
            "confidence": min(source_conf, peak),
            "source_support": source_support,
            "reversed_total": reversed_total,
            "peak": peak,
        }

    contradictory: set[tuple[str, str, int]] = set()
    for key in provisional:
        a, b, view = key
        reverse = (b, a, view)
        if reverse in provisional:
            contradictory.add(key)
            contradictory.add(reverse)
    for key in contradictory:
        provisional.pop(key, None)
        rejected["reverse_direction_contradiction"] += 1

    # Fail closed frame-by-frame if the active order graph is cyclic.
    active_by_view_frame: dict[tuple[int, int], list[tuple[str, str, int]]] = {}
    for key, row in provisional.items():
        for frame in row["frames"]:
            active_by_view_frame.setdefault((row["view"], frame), []).append(key)
    cyclic_domains: list[dict] = []
    for (view, frame), keys in active_by_view_frame.items():
        if _has_cycle((provisional[k]["front"], provisional[k]["back"]) for k in keys):
            cyclic_domains.append({"view_index": view, "frame_index": frame})
            for key in keys:
                provisional[key]["frames"].discard(frame)
                rejected["cycle_frame"] += 1

    edges: list[PresentationOrderEvidenceV1] = []
    for row in provisional.values():
        if not row["frames"]:
            rejected["empty_frame_domain"] += 1
            continue
        edges.append(
            PresentationOrderEvidenceV1(
                front_owner_id=row["front"],
                back_owner_id=row["back"],
                view_index=row["view"],
                valid_frame_indices=tuple(sorted(row["frames"])),
                confidence=float(row["confidence"]),
                source_support_count=int(row["source_support"]),
                observed_reversed_pixel_count=int(row["reversed_total"]),
                peak_reversal_fraction=float(row["peak"]),
            )
        )
    edges.sort(key=lambda x: (x.view_index, x.front_owner_id, x.back_owner_id))
    audit = {
        "schema": "RealSaS.PresentationOrderEvidenceAudit.v1",
        "status": "PASS",
        "qualified_edge_count": len(edges),
        "cyclic_domains": cyclic_domains,
        "rejections": rejected,
        "thresholds": {
            "minimum_source_confidence": minimum_source_confidence,
            "minimum_source_support": minimum_source_support,
            "minimum_total_reversed_pixels": minimum_total_reversed_pixels,
            "minimum_peak_reversal_fraction": minimum_peak_reversal_fraction,
        },
        "claims": {
            "order_is_product_qualified": False,
            "runtime_behavior_changed": False,
            "geometry_authority": "M_UNCHANGED",
        },
    }
    return edges, audit
