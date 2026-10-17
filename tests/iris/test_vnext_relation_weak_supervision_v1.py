import torch

from models.iris.vnext.training_v1 import graphics_relation_loss_v1


_KEYS = (
    "same_drawable_affinity",
    "cut_probability",
    "attachment_affinity",
    "alpha_boundary_probability",
    "appearance_boundary_probability",
    "front_probability",
    "occlusion_probability",
    "visibility_probability_a",
    "visibility_probability_b",
)


def _pred(value=0.5):
    return {key: torch.full((2,), float(value), dtype=torch.float32) for key in _KEYS}


def _truth():
    # Row 0 is a fallback adjacency row: front=0.5 is the explicit UNKNOWN sentinel.
    # Row 1 is a true two-hit ray with observed ordering/occlusion.
    return {
        "same_drawable_affinity": torch.tensor([0.0, 0.0]),
        "cut_probability": torch.tensor([1.0, 1.0]),
        "attachment_affinity": torch.tensor([0.0, 0.0]),
        "alpha_boundary_probability": torch.tensor([0.0, 1.0]),
        "appearance_boundary_probability": torch.tensor([1.0, 1.0]),
        "front_probability": torch.tensor([0.5, 1.0]),
        "occlusion_probability": torch.tensor([0.0, 1.0]),
        "visibility_probability_a": torch.tensor([1.0, 1.0]),
        "visibility_probability_b": torch.tensor([0.0, 0.0]),
    }


def test_unknown_fallback_view_relations_do_not_change_loss():
    truth = _truth()
    base = _pred()
    baseline = graphics_relation_loss_v1(base, truth)

    changed = {key: value.clone() for key, value in base.items()}
    # These row-0 targets are unknown, not negative labels.
    for key in (
        "attachment_affinity",
        "alpha_boundary_probability",
        "front_probability",
        "occlusion_probability",
        "visibility_probability_b",
    ):
        changed[key][0] = 0.999
    observed = graphics_relation_loss_v1(changed, truth)
    assert torch.allclose(baseline, observed, atol=1e-7, rtol=0.0)


def test_observed_static_relation_still_changes_loss():
    truth = _truth()
    base = _pred()
    baseline = graphics_relation_loss_v1(base, truth)
    changed = {key: value.clone() for key, value in base.items()}
    changed["cut_probability"][0] = 0.999
    observed = graphics_relation_loss_v1(changed, truth)
    assert not torch.allclose(baseline, observed, atol=1e-7, rtol=0.0)


def test_same_drawable_provides_safe_attachment_negative():
    truth = _truth()
    truth["same_drawable_affinity"][0] = 1.0
    base = _pred()
    baseline = graphics_relation_loss_v1(base, truth)
    changed = {key: value.clone() for key, value in base.items()}
    changed["attachment_affinity"][0] = 0.999
    observed = graphics_relation_loss_v1(changed, truth)
    assert not torch.allclose(baseline, observed, atol=1e-7, rtol=0.0)
