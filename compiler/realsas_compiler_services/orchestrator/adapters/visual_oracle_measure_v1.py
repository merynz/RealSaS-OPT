"""Measure oracle against raw authored cached poses; failed fit stays failed."""
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw

from compiler.realsas_compiler_core.visual_oracle_measure_v1 import (
    cached_source, independent_reference, metrics, canonical_visible, initially_occluded)
from compiler.realsas_compiler_core.visual_oracle_io_v1 import file_ref, write_json, sha
from compiler.realsas_compiler_core.runtime_package_v2 import read_rss_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload


def checked(ref):
    path = Path(ref["path"])
    if sha(path.read_bytes()) != ref["sha256"]:
        raise ValueError("ORACLE_MEASUREMENT_INPUT_DRIFT")
    return path


def measure_reference(ctx):
    evidence = stage_output_payload(ctx, "47_AUTHORED_VISUAL_ORACLE", "RealSaS.AuthoredVisualOracleEvidence.v1")
    consumed = stage_output_payload(ctx, "48_NATIVE_VISUAL_ORACLE", "RealSaS.NativeVisualOracleConsumption.v1")
    settings = ctx["run_manifest"]["oracle_measure"]
    images, reference_frames, owners = cached_source(ctx["run_manifest"]["oracle_source"])
    if len(reference_frames) != len(evidence["frames"]) or owners != sorted(evidence["owners"]):
        raise ValueError("ORACLE_REFERENCE_DOMAIN_DRIFT")
    root = ctx["run_root"] / "artifacts" / "49_VISUAL_ORACLE_MEASURE"
    root.mkdir(parents=True, exist_ok=True)
    outputs, references, initial_hidden, reports, gifs = [], [], {}, [], {}
    max_origin, max_angle, max_scale = 0.0, 0.0, 0.0
    for index, (frame, compiled) in enumerate(zip(reference_frames, evidence["frames"])):
        if (frame["clip"], frame["key_id"], frame["time_ms"]) != (compiled["clip"], compiled["key_id"], compiled["time_ms"]):
            raise ValueError("ORACLE_FRAME_ASSOCIATION_DRIFT")
        compiled_owners = {o["owner"]: o for o in compiled["objects"]}
        for obj in frame["objects"]:
            other = compiled_owners[obj["owner"]]
            if obj["member"] != other["member"] or obj["order"] != other["order"]:
                raise ValueError("ORACLE_AUTHORED_ASSET_OR_ORDER_NOT_CONSUMED")
            a, b = np.array(obj["pose"]), np.array(other["pose"])
            max_origin = max(max_origin, float(np.linalg.norm(a[:2]-b[:2])))
            max_angle = max(max_angle, float(abs((a[2]-b[2]+180)%360-180)))
            max_scale = max(max_scale, float(np.abs(a[3:5]-b[3:5]).max()))
        reference = independent_reference(frame, images, owners, evidence["resolution"], evidence["offset"])
        references.append(reference)
        if frame["clip"] not in initial_hidden:
            initial_hidden[frame["clip"]] = initially_occluded(frame, images, owners, reference[1], reference[2], evidence["offset"])
        path = root / f"reference_{index:03d}.png"; Image.fromarray(reference[0]).save(path)
        outputs.append(file_ref(path, "image/png"))
    revealed_addresses, observed_addresses = {}, {}
    resolution = evidence["resolution"]
    for row in consumed["rows"]:
        index, variant = row["frame_index"], row["variant"]
        expected, expected_owner, robust, address = references[index]
        actual = np.fromfile(checked(row["files"]["rgba"]), np.uint8).reshape(resolution, resolution, 4)
        faces = np.fromfile(checked(row["files"]["owner"]), "<i4").reshape(resolution, resolution)
        entries = read_rss_v2(checked(row["files"]["rss"]))
        face_owners = json.loads(entries["authority/oracle_face_owners.json"])
        if face_owners != row["face_owners"] or np.any(faces < -1) or np.any(faces >= len(face_owners)):
            raise ValueError("ORACLE_FACE_OWNER_RECEIPT_DRIFT")
        lookup = np.array([owners.index(o) for o in face_owners], np.int32)
        actual_owner = np.full(faces.shape, -1, np.int32)
        valid = faces >= 0; actual_owner[valid] = lookup[faces[valid]]
        report = metrics(expected, actual, expected_owner, actual_owner, robust, settings["budgets"])
        frame = reference_frames[index]
        report.update({"variant": variant, "frame_index": index, "clip": frame["clip"], "time_ms": frame["time_ms"]})
        reports.append(report)
        if variant == "baseline":
            visible = canonical_visible(frame, owners, expected_owner, address, robust)
            for key, values in visible.items():
                hidden = values & initial_hidden[frame["clip"]].get(key, set())
                cohort = (frame["clip"], *key)
                revealed_addresses.setdefault(cohort, set()).update(hidden)
                mask = robust & (expected_owner == owners.index(key[0])) & np.isin(address, list(hidden))
                e_pm, a_pm = expected.astype(float)/255, actual.astype(float)/255
                e_pm[..., :3] *= e_pm[..., 3, None]; a_pm[..., :3] *= a_pm[..., 3, None]
                material_matches = np.abs(e_pm-a_pm).max(axis=-1) <= settings["budgets"]["maximum_hidden_sample_pm_error"]
                accepted = mask & (expected_owner == actual_owner) & (actual[..., 3] >= 252) & material_matches
                observed_addresses.setdefault(cohort, set()).update(map(int, address[accepted]))
        # Composite onto a fixed checkerboard solely for viewing; metrics above
        # use raw RGBA, never the GIF's palette or presentation background.
        view = Image.new("RGB", (resolution*2, resolution + 28), (32, 34, 40))
        for col, pixels, label in ((0, expected, "Authored reference"), (1, actual, "Native: " + variant)):
            tile = Image.new("RGBA", (resolution, resolution), (220, 222, 228, 255))
            tile.alpha_composite(Image.fromarray(pixels))
            view.paste(tile.convert("RGB"), (col*resolution, 28))
            ImageDraw.Draw(view).text((col*resolution+8, 7), label, fill=(240, 240, 240))
        gifs.setdefault((variant, frame["clip"]), []).append(view)
    summaries = {}
    for variant in consumed["player"]["variants"]:
        group = [r for r in reports if r["variant"] == variant]
        summaries[variant] = {"frame_count": len(group), "passed_frames": sum(r["passed"] for r in group),
            "all_frames_passed": all(r["passed"] for r in group),
            "minimum_alpha_iou": min(r["alpha_iou"] for r in group),
            "maximum_pm_mean": max(r["pm_mean"] for r in group),
            "maximum_pm_p99": max(r["pm_p99"] for r in group),
            "robust_owner_tested_pixels": sum(r["robust_owner_tested_pixels"] for r in group),
            "robust_owner_mismatch_pixels": sum(r["robust_owner_mismatch_pixels"] for r in group),
            "failed_checks": sorted({k for r in group for k, v in r["checks"].items() if not v})}
    for (variant, clip), frames in gifs.items():
        if variant != "baseline" and clip != ctx["run_manifest"]["oracle_source"]["clips"][-1]:
            continue
        slug = "".join(c if c.isalnum() else "_" for c in clip)
        path = root / f"oracle_{variant}_{slug}.gif"
        source_frames = [f for f in reference_frames if f["clip"] == clip]
        times = [f["time_ms"] for f in source_frames] + [next(f["length_ms"] for f in evidence["frames"] if f["clip"] == clip)]
        durations = [max(20, b-a) for a, b in zip(times, times[1:])]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
        outputs.append(file_ref(path, "image/gif"))
    required = sum(len(v) for v in revealed_addresses.values())
    accepted = sum(len(v & observed_addresses.get(k, set())) for k, v in revealed_addresses.items())
    pose_pass = max_origin <= settings["budgets"]["maximum_pose_origin_px"] and max_angle <= settings["budgets"]["maximum_pose_angle_degrees"] and max_scale <= settings["budgets"]["maximum_pose_scale_error"]
    payload = {"schema": "RealSaS.AuthoredVisualOracleMeasurement.v1", "experiment_completed": True,
        "scope": evidence["scope"], "source": ctx["run_manifest"]["oracle_source"],
        "preregistered_measurement": settings, "variant_summaries": summaries, "frames": reports,
        "pose": {"maximum_origin_residual_source_pixels": max_origin, "maximum_angle_residual_degrees": max_angle,
                 "maximum_scale_residual": max_scale, "passed": pose_pass},
        "hidden_sprite_material": {"initially_occluded_then_revealed_authored_texels": required,
            "native_visible_with_correct_owner_texels": accepted, "coverage": accepted/required if required else None,
            "passed": required > 0 and accepted == required, "per_sample_PM_colour_checked": True,
            "full_anatomy_or_CAA_completion_proven": False},
        "negative_controls_detected": {v: not s["all_frames_passed"] for v, s in summaries.items() if v != "baseline"},
        "baseline_raster_fit_passed": summaries["baseline"]["all_frames_passed"] and pose_pass,
        "required_contact_cardinality": None, "required_grip_cardinality": None,
        "baked_external_PNG_timing_association_qualified": False, "Knight_3D_bind_proven": False,
        "full_visual_qualification": False, "product_revision_minted": False,
        "not_measured": ["visual_contact_truth", "hand_prop_grip_truth", "3D_to_2D_bind", "unseen/generalization", "CAA_completion_transport", "continuous_interframe_motion"]}
    outputs.append(write_json(root / "measurement.json", payload))
    # Stage PASS means exact experiment output exists, never visual fit PASS.
    return {"status": "PASS", "outputs": outputs, "diagnostics": {
        "experiment_completed": True, "baseline_raster_fit_passed": payload["baseline_raster_fit_passed"],
        "hidden_sprite_material": payload["hidden_sprite_material"], "negative_controls_detected": payload["negative_controls_detected"],
        "full_visual_qualification": False}}
