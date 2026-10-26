"""Truth Foundry V1 envelope validation, independent of teacher authoring names.

This is local structural validation. Registry admission, student observability
qualification and product promotion are separate Platform/Compiler decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re


SCHEMA = "RealSaS.TruthPackageManifest.v1"
FAMILIES = frozenset({
    "SURFACE_TRAJECTORY", "SURFACE_METRIC_STRAIN", "RELATIVE_AFFINE_TRANSPORT",
    "ORIENTED_GAP_AND_VELOCITY", "ADMISSIBLE_MOTION_SUBSPACE",
    "PHYSICAL_IMPULSE_RESPONSE", "RAY_COMPOSITING", "RASTER_EDIT_RESPONSE",
    "BOUNDARY_RESPONSE", "TEMPORAL_RESPONSE",
})
OBSERVABILITY = frozenset({"OBSERVATION_CONDITIONED", "MECHANISM_CONDITIONAL",
                          "PRIVILEGED_ONLY", "AMBIGUOUS", "UNKNOWN"})


def sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("lowercase SHA256 required")
    return value


def exact_keys(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError("unexpected or missing keys: " + repr(required))


@dataclass(frozen=True)
class SourceLocusRefV1:
    source_sha256: str
    topology_sha256: str
    primitive_index: int
    face_index: int
    barycentric: tuple[float, float, float]

    def __post_init__(self):
        sha256(self.source_sha256)
        sha256(self.topology_sha256)
        if any(type(v) is not int or v < 0 for v in (self.primitive_index, self.face_index)):
            raise ValueError("source primitive/face indices must be nonnegative integers")
        b = self.barycentric
        if (len(b) != 3 or any(not math.isfinite(x) or x < 0 or x > 1 for x in b)
                or not math.isclose(sum(b), 1, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("valid barycentric coordinates required")


def validate_manifest(value):
    """Validate exact wire envelope; never upgrades a measurement to truth PASS."""
    exact_keys(value, {"schema", "subject_key", "source_sha256", "topology_sha256",
                       "teacher_scene_sha256", "intervention_spec_sha256",
                       "observation_bundle_sha256", "producer_source_commit",
                       "coordinate_contract", "families", "payloads", "authority"})
    if value["schema"] != SCHEMA or value["authority"] != "UNQUALIFIED_DERIVED_MEASUREMENTS":
        raise ValueError("writer cannot mint qualification authority")
    if not isinstance(value["subject_key"], str) or not value["subject_key"].strip():
        raise ValueError("nonempty subject key required")
    for key in ("source_sha256", "topology_sha256", "teacher_scene_sha256",
                "intervention_spec_sha256", "observation_bundle_sha256"):
        sha256(value[key])
    if not re.fullmatch(r"[0-9a-f]{40}", value["producer_source_commit"]):
        raise ValueError("exact source commit required")
    coord = value["coordinate_contract"]
    exact_keys(coord, {"frame", "length_unit", "seconds_per_time_unit", "normalization_sha256"})
    if coord["frame"] != "SOURCE_NORMALIZED" or coord["length_unit"] not in ("METRE", "SOURCE_UNIT"):
        raise ValueError("explicit supported source coordinate contract required")
    if type(coord["seconds_per_time_unit"]) not in (float, int) or not math.isfinite(coord["seconds_per_time_unit"]) or coord["seconds_per_time_unit"] <= 0:
        raise ValueError("positive time conversion required")
    sha256(coord["normalization_sha256"])
    if not isinstance(value["families"], list) or not value["families"]:
        raise ValueError("explicit family coverage required")
    seen = set()
    for family in value["families"]:
        exact_keys(family, {"family", "measurement_status", "observability", "witness_sha256",
                            "validity_array", "reason", "payload_arrays"})
        name = family["family"]
        if name not in FAMILIES or name in seen:
            raise ValueError("unknown or duplicate measurement family")
        seen.add(name)
        status, obs = family["measurement_status"], family["observability"]
        if status not in ("MEASURED", "UNAVAILABLE", "UNSUPPORTED") or obs not in OBSERVABILITY:
            raise ValueError("explicit measurement/observability state required")
        if not isinstance(family["reason"], str) or not family["reason"]:
            raise ValueError("coverage/abstention reason required")
        if family["witness_sha256"] is not None:
            sha256(family["witness_sha256"])
        if obs in ("OBSERVATION_CONDITIONED", "MECHANISM_CONDITIONAL") and family["witness_sha256"] is None:
            raise ValueError("conditioning/observability witness required")
        arrays = family["payload_arrays"]
        if not isinstance(arrays, list) or any(not isinstance(a, str) for a in arrays) or len(set(arrays)) != len(arrays):
            raise ValueError("unique array names required")
        if status == "MEASURED":
            if not arrays or not isinstance(family["validity_array"], str) or not family["validity_array"]:
                raise ValueError("measured family requires payload and validity mask")
        elif arrays or family["validity_array"] is not None or obs not in ("UNKNOWN", "PRIVILEGED_ONLY"):
            raise ValueError("unavailable family cannot carry measured/supervised payload")
    if seen != FAMILIES:
        raise ValueError("every V1 family must explicitly report coverage or abstention")
    if not isinstance(value["payloads"], dict):
        raise ValueError("array descriptor mapping required")
    for name, desc in value["payloads"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("portable array name required")
        exact_keys(desc, {"path", "sha256", "dtype", "shape"})
        if desc["path"] != "arrays/" + name + ".npy":
            raise ValueError("array path cannot escape package")
        sha256(desc["sha256"])
        if not isinstance(desc["dtype"], str) or not re.fullmatch(r"[<|][biuf][1248]", desc["dtype"]):
            raise ValueError("little-endian numeric/bool dtype required")
        if not isinstance(desc["shape"], list) or any(type(x) is not int or x < 0 for x in desc["shape"]):
            raise ValueError("nonnegative integer dimensions required")
    for family in value["families"]:
        for name in family["payload_arrays"] + ([family["validity_array"]] if family["validity_array"] else []):
            if name not in value["payloads"]:
                raise ValueError("family references missing payload")
        if family["validity_array"] and value["payloads"][family["validity_array"]]["dtype"] != "|b1":
            raise ValueError("validity mask must be boolean")
    return value
