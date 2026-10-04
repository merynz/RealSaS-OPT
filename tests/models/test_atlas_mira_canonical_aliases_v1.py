from models.atlas import (
    ATLASReferenceStrengthCandidateV1,
    ATLASReferenceStrengthConfigV1,
    ATLASSurfaceEncoderV1,
)
from models.mira import (
    MIRABackboneV4,
    MIRABackboneConfigV4,
    MIRACandidateV5,
    MIRACandidateConfigV5,
)


def test_atlas_aliases_preserve_legacy_types():
    cfg = ATLASReferenceStrengthConfigV1()
    model = ATLASReferenceStrengthCandidateV1(cfg)
    assert isinstance(model.encoder, ATLASSurfaceEncoderV1)
    assert "Geppetto" in cfg.architecture_id  # legacy checkpoint identity intentionally preserved


def test_mira_aliases_preserve_legacy_types():
    bcfg = MIRABackboneConfigV4()
    backbone = MIRABackboneV4(bcfg)
    mcfg = MIRACandidateConfigV5()
    model = MIRACandidateV5(backbone=backbone, config=mcfg)
    assert model.backbone is backbone
    assert "Arachne" in bcfg.architecture_id  # legacy checkpoint identity intentionally preserved
