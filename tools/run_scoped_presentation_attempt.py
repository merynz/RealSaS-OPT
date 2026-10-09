"""Run the explicit reduced presentation DAG through Go, then export its evidence."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler.realsas_compiler_services.orchestrator import mainline
from tools.platform_release_snapshot import snapshot

ACTOR = "authorized-knight-presentation-p0"


def research_plan(variant="v3", relational_gate=False):
    plan = deepcopy(mainline.load_json(mainline.PLAN_PATH))
    selected = (37, 42, 43, 44, 45)
    functions = ("compile_source_domains_stage", "compile_projection_stage", "package_stage",
                 "playback_stage", "prove_presentation_stage")
    stages = [deepcopy(plan["stages"][i-1]) for i in selected]
    ids = [s["id"] for s in stages]
    dependencies = ([], [ids[0]], [ids[1]], [ids[1], ids[2]], ids[:4])
    for ordinal, (stage, function, deps) in enumerate(zip(stages, functions, dependencies), 1):
        stage.update(ordinal=ordinal, depends_on=deps,
            adapter="compiler.realsas_compiler_services.orchestrator.adapters.presentation_research_" + (
                "v1" if relational_gate and ordinal in (1, 3, 4) else
                "v4" if relational_gate and ordinal == 5 else variant) + ":" + function,
            manifest_keys=["presentation_research"] + (["runtime"] if ordinal >= 4 else []))
        stage["policy"]["product_pass_authority"] = False
    plan.update(stage_count=len(stages), stages=stages)
    mainline.validate_plan(plan, require_product_pass_authority=False)
    return plan


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    def ctl(operation, value=None, extra=()):
        command = [str(args.ctl), operation, "--api", args.api, *extra]
        if value is not None:
            path = args.out / (operation + "-request.json")
            path.write_text(json.dumps(value, indent=2) + "\n")
            command += ["--request", str(path)]
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=60)
        return json.loads(result.stdout)

    cfg = json.loads(args.config.read_text())
    # Fresh host configuration references immutable bytes; imported notebook
    # payloads and their embedded provenance are never rewritten.
    def materialize(value):
        if isinstance(value, list):
            return [materialize(v) for v in value]
        if isinstance(value, dict):
            value = {k: materialize(v) for k, v in value.items()}
            if "path" in value and "sha256" in value:
                path = args.input_root / Path(value["path"]).name
                if hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]:
                    raise RuntimeError("SCOPED_INPUT_BYTES_DRIFT:" + path.name)
                value["path"] = str(path.resolve())
            return value
        return value
    cfg = materialize(cfg)
    handoff = json.loads(Path(cfg["source_handoff"]["path"]).read_text())
    for view in cfg["views"]:
        source = view["source"]
        receipt = handoff["files"][f"observations/V{view['view_index']}.png"]
        if source["sha256"] != receipt["sha256"] or Path(source["path"]).stat().st_size != receipt["bytes"]:
            raise RuntimeError("SCOPED_SOURCE_HANDOFF_BYTES_DRIFT")
    player = args.player.resolve()
    runtime = {"native_player": {"path": str(player), "sha256": hashlib.sha256(player.read_bytes()).hexdigest()},
               "native_parallel_workers": 2}
    run_id = "KNIGHT_PRESENTATION_P0_" + uuid.uuid4().hex.upper()
    manifest = {"run_id": run_id, "subject_id": "KNIGHT_FIT1_SCOPED_PRESENTATION",
                "execution_class": "DEMO_WITNESS", "presentation_research": cfg, "runtime": runtime}
    manifest_path = mainline.run_manifest_path(run_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    subject = getattr(args, "subject_id", None) or ctl("subject", {"slug": "knight-fit1-scoped-presentation", "display_name": "Knight sealed FIT1 presentation",
                              "created_by": ACTOR})["subject_id"]
    if getattr(args, "reuse_inputs", None):
        inputs = {"subject_input_id": args.reuse_inputs["subject_input_id"]}
        input_artifacts = args.reuse_inputs["input_artifacts"]
    else:
        bindings = []
        input_artifacts = []
        def refs(value):
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    yield value
                else:
                    for child in value.values(): yield from refs(child)
            elif isinstance(value, list):
                for child in value: yield from refs(child)
        seen = set()
        for ref in refs(cfg):
            if ref["sha256"] in seen: continue
            seen.add(ref["sha256"])
            obj = ctl("artifact-put", extra=("--file", ref["path"], "--sha256", ref["sha256"]))
            role = "sealed-input:" + Path(ref["path"]).name
            artifact = ctl("artifact-import", {"artifact_type": "RealSaS.SealedPresentationSourceBytes",
                "schema_version": "v1", "object": obj, "source_uri": "urn:sha256:" + ref["sha256"], "created_by": ACTOR})
            bindings.append({"role": role, "artifact_id": artifact["artifact_id"]})
            input_artifacts.append({"role": role, "artifact_id": artifact["artifact_id"], "sha256": ref["sha256"]})
        for key, section in (("presentation_research", cfg), ("runtime", runtime)):
            path = args.out / (key + ".json")
            path.write_text(json.dumps(section, sort_keys=True) + "\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            obj = ctl("artifact-put", extra=("--file", str(path), "--sha256", digest))
            artifact = ctl("artifact-import", {"artifact_type": "RealSaS.ScopedPresentationInput",
                "schema_version": "v1", "object": obj, "source_uri": "urn:realsas:sealed-fit:presentation:" + key,
                "created_by": ACTOR})
            bindings.append({"role": "manifest:" + key, "artifact_id": artifact["artifact_id"]})
            input_artifacts.append({"role": key, "artifact_id": artifact["artifact_id"], "sha256": digest})
        inputs = ctl("input-seal", {"subject_id": subject, "bindings": bindings, "created_by": ACTOR})
    plan = research_plan(getattr(args, "variant", "v3"), getattr(args, "relational_gate", False))
    release_request = snapshot(plan, manifest, name="knight-presentation-p0-" + run_id,
                               purpose="RESEARCH", created_by=ACTOR)
    release = ctl("release", release_request)["ReleaseID"]
    start_request = {"subject_id": subject, "baseline_engine_release_id": getattr(args, "baseline_release", None) or release,
        "candidate_engine_release_id": release, "created_by": ACTOR}
    if getattr(args, "parent_attempt", None): start_request["parent_attempt_id"] = args.parent_attempt
    attempt = ctl("research-start", start_request)
    (args.out / "research_start.json").write_text(json.dumps(attempt, indent=2) + "\n")
    if getattr(args, "require_stage37_reuse", False):
        changed = [row["StageID"] for row in attempt["Impact"]["DirectChanges"]]
        if changed != [plan["stages"][1]["id"]]:
            raise RuntimeError("MATCHED_PRESENTATION_INTERVENTION_SCOPE_DRIFT:" + repr(changed))
    attempt_id = attempt["AttemptID"]
    scope = {"attempt_id": attempt_id, "engine_release_id": release, "compiler_run_id": run_id,
             "code_sha": os.environ.get("GITHUB_SHA", "LOCAL_UNPUBLISHED"),
             "target_stage_id": plan["stages"][-1]["id"],
             "requested_target_closure": [s["id"] for s in plan["stages"]],
             "code_change_impact": attempt["Impact"],
             "reused_sealed_inputs": {k: cfg[k] for k in ("weights", "axis", "alignment", "frame_qualification", "frame_metrics", "motion")},
             "mechanical_model_stages_in_graph": [], "product_authority": False, "input_artifacts": input_artifacts}
    (args.out / "resolved_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print(json.dumps({"attempt_id": attempt_id, "target": scope["target_stage_id"],
                      "requested_target_closure": scope["requested_target_closure"], "mechanics": "REUSE_EXACT_SEALED_INPUTS"}), flush=True)
    ctl("research-run", {"research_attempt_id": attempt_id, "subject_id": subject, "engine_release_id": release,
        "subject_input_id": inputs["subject_input_id"], "target_stage_id": scope["target_stage_id"],
        "compiler_run_id": run_id, "run_manifest_path": str(manifest_path),
        "run_ledger_path": str(mainline.run_ledger_path(run_id)), "pipeline_plan_sha256": mainline.content_sha256(plan),
        "idempotency_key": "presentation-p0:" + run_id, "requested_by": ACTOR})
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        state = ctl("attempt", extra=("--id", attempt_id))
        if state["state"] != "OPEN":
            break
        time.sleep(2)
    else:
        raise RuntimeError("SCOPED_ATTEMPT_TIMEOUT:" + attempt_id)
    (args.out / "attempt_state.json").write_text(json.dumps(state, indent=2) + "\n")
    ledger = mainline.load_json(mainline.run_ledger_path(run_id))
    (args.out / "execution_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    scope["resolved_stage_statuses"] = [{"stage_id": s["id"], "status": s["status"]} for s in ledger["stages"]]
    scope["execute_scope"] = [s["id"] for s in ledger["stages"]
                              if not s.get("platform_artifact_id") and s["status"] not in ("PENDING", "BLOCKED")]
    scope["reuse_scope"] = [s["id"] for s in ledger["stages"] if s.get("platform_artifact_id")]
    (args.out / "resolved_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    if getattr(args, "require_stage37_reuse", False) and plan["stages"][0]["id"] not in scope["reuse_scope"]:
        raise RuntimeError("MATCHED_PRESENTATION_STAGE37_EXACT_REUSE_NOT_PROVEN")
    # Retain independent typed stage evidence even when a later gate fails.
    # Inputs, transport keys and private locators are outside this tree.
    artifacts = mainline.authority_root() / "runs" / run_id / "artifacts"
    for stage in ledger["stages"]:
        for output in stage.get("outputs", ()):
            path = Path(output["path"])
            if output["schema"] == "application/x-rgba8" or not path.is_file(): continue
            target = args.out / "stage_artifacts" / path.relative_to(artifacts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    proof = mainline.authority_root() / "runs" / run_id / "artifacts" / scope["target_stage_id"] / "presentation_proof.json"
    if proof.is_file():
        from PIL import Image, ImageDraw
        value = json.loads(proof.read_text())
        (args.out / "presentation_proof.json").write_bytes(proof.read_bytes())
        for clip in ("demo_idle_v1", "demo_run_v1", "demo_slash_v1"):
            for vi in range(8):
                rows = [r for r in value["rendered_frames"] if r["clip_id"] == clip and r["view_id"] == f"V{vi}"]
                frames = []
                for row in rows:
                    resolution = int(cfg.get("render_resolution", 256))
                    raw = Path(row["rgba"]["path"]).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != row["rgba"]["sha256"]:
                        raise RuntimeError("SCOPED_EXPORT_FRAME_BYTES_DRIFT")
                    frame = Image.frombytes("RGBA", (resolution, resolution), raw)
                    background = Image.new("RGBA", frame.size, (34, 38, 46, 255))
                    background.alpha_composite(frame)
                    frames.append(background.convert("RGB").resize((768, 768), Image.Resampling.NEAREST))
                if frames:
                    label = clip.split("_")[1].upper()
                    path = args.out / f"Knight_{label}_V{vi}_768.gif"
                    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000/24, loop=0)
        if value["status"] != "PASS_DEMO_ONLY" and not getattr(args, "allow_proof_fail", False):
            raise RuntimeError("SCOPED_PRESENTATION_PROOF_FAILED__EXPORTED_DIAGNOSTICS_ONLY")
    elif getattr(args, "allow_proof_fail", False):
        raise RuntimeError("MATCHED_PRESENTATION_PROOF_NOT_PRODUCED")
    if state["state"] != "COMPLETED" and not getattr(args, "allow_proof_fail", False):
        raise RuntimeError("SCOPED_ATTEMPT_FAILED:" + attempt_id)
    receipt = {"subject_id": subject, "subject_input_id": inputs["subject_input_id"],
        "input_artifacts": input_artifacts, "engine_release_id": release, "attempt_id": attempt_id,
        "compiler_run_id": run_id, "attempt_state": state["state"], "proof_status": value["status"] if proof.is_file() else None}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--ctl", type=Path, required=True)
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=int, default=1800)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
