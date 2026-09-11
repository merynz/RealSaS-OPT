from __future__ import annotations

"""Frozen hash/locator authority for the promoted Mage FIT1 Arachne V5 composite checkpoint."""

PROMOTION_COMMIT = "03d9f87dbb7100a72293915cf682cbf338335a37"
ARCHITECTURE_ID = "RealSaS.Arachne.A1.MinimalK4DirectSimplex.v5"
V5_SOURCE_SHA256 = "a66adaebe92e9181873888ef90941ad87e4b3d835d21647e65176df4441a0f9e"

BACKBONE_CHECKPOINT_SHA256 = "95c441f97b02123de1a5bc83bdf5ad223363c4b97927e8d420a0d246efbc1763"
BACKBONE_CHECKPOINT_DRIVE_ID = "1HJC4GPMcWa0jEXCYgctUtW09tXn4I9ND"
BACKBONE_SOURCE_COMMIT = "80bdde251f6c9d9f2c026c1a647b6192bd09d5e2"
BACKBONE_BEST_STEP = 10752

DECODER_DELTA_SHA256 = "13344178bf1b3ce96c9356456db0ad2c8a3945182a5ec63617c50137b8c52137"
DECODER_DELTA_DRIVE_ID = "1_o2XWO5BOSDKGvewJDSGS16jlcMuT9Dx"
DECODER_PARENT_SHA256 = "078d5155f798d7b926c19463a31bf01c19973be30787f5fbff7916be87894832"
DECODER_PARAMETER_COUNT = 325_313
TOTAL_PARAMETER_COUNT = 138_378_466

COMPOSITE_MANIFEST_SHA256 = "a6fec97b84739452e4b0126c4107a54f23a585d2f14c00548e0501d12b4c9a4f"
CLOSURE_REPORT_SHA256 = "6ba5d63a5e9d5cab0ebc6f374bc31e86ffb2a3ed325b89dc9880b5b858ce83f4"
CLOSURE_SEAL_SHA256 = "11fad459db94bc5604585fb63f22a018f538c8d30084d92c0c4e83747c619048"
FIT1_GSA_P95 = 0.04237784981177733
FIT1_DEFORMATION_RATIO = 0.019856400787830353
FIT1_ARTICULATED_DEFORMATION_RATIO = 0.002780771814286709
GENERALIZATION_CLAIM = False
PRODUCT_PASS_CLAIM = False


def sealed_arachne_v5_fit1_authority() -> dict[str, object]:
    return {
        "promotion_commit": PROMOTION_COMMIT,
        "architecture_id": ARCHITECTURE_ID,
        "v5_source_sha256": V5_SOURCE_SHA256,
        "backbone_checkpoint_sha256": BACKBONE_CHECKPOINT_SHA256,
        "backbone_checkpoint_drive_id": BACKBONE_CHECKPOINT_DRIVE_ID,
        "backbone_source_commit": BACKBONE_SOURCE_COMMIT,
        "backbone_best_step": BACKBONE_BEST_STEP,
        "decoder_delta_sha256": DECODER_DELTA_SHA256,
        "decoder_delta_drive_id": DECODER_DELTA_DRIVE_ID,
        "decoder_parent_sha256": DECODER_PARENT_SHA256,
        "decoder_parameter_count": DECODER_PARAMETER_COUNT,
        "total_parameter_count": TOTAL_PARAMETER_COUNT,
        "composite_manifest_sha256": COMPOSITE_MANIFEST_SHA256,
        "closure_report_sha256": CLOSURE_REPORT_SHA256,
        "closure_seal_sha256": CLOSURE_SEAL_SHA256,
        "fit1_gsa_p95": FIT1_GSA_P95,
        "fit1_deformation_ratio": FIT1_DEFORMATION_RATIO,
        "fit1_articulated_deformation_ratio": FIT1_ARTICULATED_DEFORMATION_RATIO,
        "generalization_claim": GENERALIZATION_CLAIM,
        "product_pass_claim": PRODUCT_PASS_CLAIM,
    }
