from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import run_arachne_mage_a0_fs1_v2 as fs1
import run_arachne_mage_a0_historical_1cc449 as hist
from models.skin_field_codec.v2.config_v2 import EXPECTED_SKIN_FIELD_CODEC_V2_CONFIG_HASH
from models.skin_field_codec.v2.skin_field_codec_v2 import SkinFieldCodecV2, skin_field_codec_loss_v2


SCHEMA = "RealSaS.ArachneMageA0PairGeometryRun.v2"
EXPECTED_PREREG_GIT_BLOB = "4c0938b3771ae142314851c2a3870411a907fbee"
EXPECTED_FS1_WRAPPER_GIT_BLOB = "00ccbd7d79707fa6dab5329baeebac20efb8f470"
EXPECTED_CODEC_CONFIG_GIT_BLOB = "b12d4473b7239d9b96b3c0a963926fcb4fb61762"
EXPECTED_CODEC_SOURCE_GIT_BLOB = "815aacbeebe7686b1da4faf148a7b7013899dbaf"
EXPECTED_PAIR_GEOMETRY_SHA = "416d5e5848ec579eb67e5e402dbf6e60908f4d9a30dbe0650a1d19be7183e4e7"
EXPECTED_PAIR_GEOMETRY_SHAPE = (950, 22, 10)
EXPECTED_PARAMETER_COUNT = 414_146


@dataclass(frozen=True)
class V2Tensors:
    surface_features: torch.Tensor
    joint_features: torch.Tensor
    teacher_weights: torch.Tensor
    full_surface_mask: torch.Tensor
    teacher_surface_mask: torch.Tensor
    joint_mask: torch.Tensor
    rest_points_world: torch.Tensor
    pair_geometry: torch.Tensor

    def legacy_tuple(self):
        return (
            self.surface_features,
            self.joint_features,
            self.teacher_weights,
            self.full_surface_mask,
            self.teacher_surface_mask,
            self.joint_mask,
            self.rest_points_world,
        )

    def __iter__(self):
        return iter(self.legacy_tuple())

    def __len__(self):
        return 7

    def __getitem__(self, index):
        return self.legacy_tuple()[index]


def _count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _assert_source_authority(prereg: Path) -> None:
    if fs1.git_blob_sha(prereg) != EXPECTED_PREREG_GIT_BLOB:
        raise RuntimeError("V2_PREREG_GIT_BLOB_DRIFT")
    if fs1.git_blob_sha(Path(fs1.__file__).resolve()) != EXPECTED_FS1_WRAPPER_GIT_BLOB:
        raise RuntimeError("FS1_WRAPPER_GIT_BLOB_DRIFT")

    import models.skin_field_codec.v2.config_v2 as config_module
    import models.skin_field_codec.v2.skin_field_codec_v2 as codec_module

    if fs1.git_blob_sha(Path(config_module.__file__).resolve()) != EXPECTED_CODEC_CONFIG_GIT_BLOB:
        raise RuntimeError("V2_CODEC_CONFIG_GIT_BLOB_DRIFT")
    if fs1.git_blob_sha(Path(codec_module.__file__).resolve()) != EXPECTED_CODEC_SOURCE_GIT_BLOB:
        raise RuntimeError("V2_CODEC_SOURCE_GIT_BLOB_DRIFT")

    prereg_data = json.loads(prereg.read_text(encoding="utf-8"))
    if prereg_data.get("schema") != "RealSaS.ArachneMageA0PairGeometryCodecV2Prereg.v1":
        raise RuntimeError("V2_PREREG_SCHEMA_DRIFT")
    if int(prereg_data["frozen_architecture"]["v2_trainable_parameter_count"]) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("V2_PREREG_PARAMETER_COUNT_DRIFT")
    if int(prereg_data["single_scientific_intervention"]["decoder_input_dim"]) != 512:
        raise RuntimeError("V2_PREREG_DECODER_WIDTH_DRIFT")
    if prereg_data["single_scientific_intervention"].get("teacher_encoder_pair_geometry_access") is not False:
        raise RuntimeError("V2_PREREG_TEACHER_GEOMETRY_POLICY_DRIFT")
    if int(prereg_data.get("main_v2_scientific_optimizer_steps_at_prereg", -1)) != 0:
        raise RuntimeError("V2_PREREG_OPTIMIZER_HISTORY_DRIFT")


def patch_pair_geometry_v2(cache_manifest: Path, prereg: Path) -> None:
    _assert_source_authority(prereg)

    # Reuse the already-sealed FS1 authority repair. This preserves cache,
    # teacher, target, optimizer, scheduler, gates and resume semantics.
    fs1.SCHEMA = SCHEMA
    fs1.patch_historical_runner(cache_manifest)

    original_tensors = hist.tensors

    def tensors_v2(cache: dict[str, np.ndarray], device: torch.device) -> V2Tensors:
        if "pair_geometry" not in cache:
            raise RuntimeError("PAIR_GEOMETRY_MISSING_FROM_SEALED_CACHE")
        pg_np = np.asarray(cache["pair_geometry"], dtype=np.float32)
        if pg_np.shape != EXPECTED_PAIR_GEOMETRY_SHAPE:
            raise RuntimeError("PAIR_GEOMETRY_SHAPE_DRIFT")
        if not np.isfinite(pg_np).all():
            raise RuntimeError("PAIR_GEOMETRY_NONFINITE")
        if fs1.raw_array_sha(pg_np) != EXPECTED_PAIR_GEOMETRY_SHA:
            raise RuntimeError("PAIR_GEOMETRY_CONTENT_SHA_DRIFT")
        sf, jf, w, full_sm, teacher_sm, jm, rest = original_tensors(cache, device)
        pg = torch.tensor(pg_np[None], dtype=torch.float32, device=device)
        return V2Tensors(sf, jf, w, full_sm, teacher_sm, jm, rest, pg)

    def decode_v2(codec: SkinFieldCodecV2, tensors: V2Tensors) -> torch.Tensor:
        sf, jf, w, full_sm, teacher_sm, jm, rest = tensors
        lat = codec.encode_teacher_weights(sf, jf, w, teacher_sm, jm)
        pred, _ = codec.decode_from_latents(lat, sf, jf, tensors.pair_geometry, full_sm, jm)
        return pred

    def train_step_v2(codec: SkinFieldCodecV2, opt, tensors: V2Tensors, transforms):
        sf, jf, w, full_sm, teacher_sm, jm, rest = tensors
        codec.train()
        opt.zero_grad(set_to_none=True)
        lat = codec.encode_teacher_weights(sf, jf, w, teacher_sm, jm)
        pred, _ = codec.decode_from_latents(lat, sf, jf, tensors.pair_geometry, full_sm, jm)
        rec = skin_field_codec_loss_v2(pred, w, teacher_sm, jm)
        deform = hist.codec_deformation_loss_v1(pred, w, rest, transforms, teacher_sm, jm)
        total = rec["total"] + deform["deformation_mse"]
        total.backward()
        opt.step()
        return {
            "total": float(total.detach().cpu()),
            "reconstruction": float(rec["total"].detach().cpu()),
            "deformation_mse": float(deform["deformation_mse"].detach().cpu()),
        }

    def proposal_from_matrix_v2(mat, sids, jids, surface, sk):
        influences = []
        a = np.asarray(mat, np.float64)
        for i, sid in enumerate(sids):
            for j, jid in enumerate(jids):
                influences.append(hist.SkinInfluenceProposal(sid, jid, float(a[i, j])))
        return hist.SkinProposalIR(
            tuple(influences),
            surface.geometry_lineage_hash,
            sk.skeleton_lineage_hash,
            model_provenance="ARACHNE_MAGE_A0_PAIR_GEOMETRY_CODEC_V2",
        )

    hist.SkinFieldCodecV1 = SkinFieldCodecV2
    hist.skin_field_codec_loss_v1 = skin_field_codec_loss_v2
    hist.EXPECTED_CODEC_HASH = EXPECTED_SKIN_FIELD_CODEC_V2_CONFIG_HASH
    hist.tensors = tensors_v2
    hist.decode = decode_v2
    hist.train_step = train_step_v2
    hist.proposal_from_matrix = proposal_from_matrix_v2

    probe = SkinFieldCodecV2()
    if probe.config.config_hash != EXPECTED_SKIN_FIELD_CODEC_V2_CONFIG_HASH:
        raise RuntimeError("V2_CODEC_CONFIG_HASH_DRIFT")
    if _count_trainable(probe) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("V2_CODEC_PARAMETER_COUNT_DRIFT")
    if probe.decoder.net[0].in_features != 512:
        raise RuntimeError("V2_CODEC_DECODER_WIDTH_DRIFT")
    if probe.pair_geometry_embed[0].in_features != 10 or probe.pair_geometry_embed[0].out_features != 64:
        raise RuntimeError("V2_PAIR_GEOMETRY_PROJECTION_DRIFT")


def _augment_json(path: Path, extra: dict) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(extra)
    fs1.write_json(path, data)


def augment_outputs(output_dir: Path) -> None:
    common = {
        "architecture_id": "RealSaS.SkinFieldCodec.ContinuousJointField.PairGeometry.v2",
        "codec_config_hash": EXPECTED_SKIN_FIELD_CODEC_V2_CONFIG_HASH,
        "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
        "pair_geometry_raw_sha256": EXPECTED_PAIR_GEOMETRY_SHA,
        "pair_geometry_shape": list(EXPECTED_PAIR_GEOMETRY_SHAPE),
        "pair_geometry_projection": "Linear(10,64)+GELU",
        "decoder_input_dim": 512,
        "teacher_encoder_pair_geometry_access": False,
        "v1_warm_start": False,
    }
    _augment_json(output_dir / "ARACHNE_MAGE_A0_PREFLIGHT.json", common)
    _augment_json(output_dir / "ARACHNE_MAGE_A0_FS1_RUNNER_REGRESSION.json", common)
    _augment_json(output_dir / "ARACHNE_MAGE_A0_RESULT.json", common)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--cache-manifest", type=Path, required=True)
    ap.add_argument("--prereg", type=Path, required=True)
    ap.add_argument("--zero-surface", type=Path, required=True)
    ap.add_argument("--camera-dir", type=Path, required=True)
    ap.add_argument("--qualified-skeleton", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--regression-only", action="store_true")
    args = ap.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preflight_only and args.regression_only:
        raise RuntimeError("CHOOSE_ONE_DISPOSABLE_MODE")

    patch_pair_geometry_v2(args.cache_manifest, args.prereg)

    if args.regression_only:
        report = fs1.run_regression(
            args.cache,
            args.zero_surface,
            args.camera_dir,
            args.qualified_skeleton,
            args.output_dir,
        )
        augment_outputs(args.output_dir)
        print("A0_PAIR_GEOMETRY_V2_REGRESSION=" + json.dumps(report, sort_keys=True), flush=True)
        return 0

    forwarded = [
        "--cache", str(args.cache),
        "--zero-surface", str(args.zero_surface),
        "--camera-dir", str(args.camera_dir),
        "--qualified-skeleton", str(args.qualified_skeleton),
        "--output-dir", str(args.output_dir),
    ]
    if args.require_cuda:
        forwarded.append("--require-cuda")
    if args.preflight_only:
        forwarded.append("--preflight-only")

    rc = hist.main(forwarded)
    if not args.preflight_only:
        fs1.repair_result_metadata(args.output_dir)
    augment_outputs(args.output_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
