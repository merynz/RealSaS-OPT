from models.axis.axis_bfs_discrete_xyz_v541 import (
    AXISBFSDiscreteXYZV541,
    AXIS_COORD_BINS,
    ForbiddenRuntimeDiffusionV541,
)


def test_v541_training_authority_freezes_everything_except_coordinate_delta():
    model=AXISBFSDiscreteXYZV541().retire_legacy_diffusion_runtime().configure_v541_training_authority()
    names=model.v541_trainable_parameter_names()
    assert names
    assert all(name.startswith("coord_token_delta.") for name in names)
    assert isinstance(model.diffusion,ForbiddenRuntimeDiffusionV541)
    assert AXIS_COORD_BINS==256


def test_v541_proposal_contract_source_declares_no_parent_reselection_or_diffusion():
    import inspect
    source=inspect.getsource(AXISBFSDiscreteXYZV541.propose)
    assert '"compiler_role":"VALIDATE_AND_MATERIALIZE"' in source
    assert '"compiler_parent_reselection_allowed":False' in source
    assert '"geometry_authority":"ORDERED_256_BIN_XYZ_ARGMAX"' in source
    assert '"diffusion_runtime_authority":False' in source
