import numpy as np

from compiler.realsas_compiler_core.proof_engine import mutation_worsens_measurement
from compiler.realsas_compiler_services.numerics.lbs import (
    apply_lbs_probe_v1,
    lbs_probe_report_v1,
)
from compiler.realsas_compiler_services.proof.causal_mutations import swap_weight_mass_v1


def test_verified_lbs_weight_mutation_is_causal():
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64)
    w = np.array([[1.0, 0.0], [0.0, 1.0]], np.float64)
    transforms = np.tile(np.eye(4), (1, 2, 1, 1))
    transforms[0, 1, 0, 3] = 1.0
    expected = apply_lbs_probe_v1(p, w, transforms)
    base = lbs_probe_report_v1(apply_lbs_probe_v1(p, w, transforms), expected)
    mutated_weights = swap_weight_mass_v1(w, joint_a=0, joint_b=1, fraction=0.5)
    bad = lbs_probe_report_v1(apply_lbs_probe_v1(p, mutated_weights, transforms), expected)
    assert base.rms == 0.0 and bad.rms > base.rms and bad.p95 > base.p95
    assert mutation_worsens_measurement(
        "DEFORMATION",
        {"rms": base.rms, "p95": base.p95},
        {"rms": bad.rms, "p95": bad.p95},
    )


def test_all_domain_mutation_comparators_are_causal():
    cases = {
        "MECHANICAL_STRUCTURE": (
            {"illegal_parent_count": 0, "deform_root_count": 1, "unsupported_joint_count": 0},
            {"illegal_parent_count": 0, "deform_root_count": 1, "unsupported_joint_count": 1},
        ),
        "MESH_QUALITY": (
            {"degenerate_faces": 0, "min_area": 0.1},
            {"degenerate_faces": 1, "min_area": 0.0},
        ),
        "DIRECTIONAL_VISUAL": (
            {"direction_count": 8, "corner_binding_count": 24},
            {"direction_count": 7, "corner_binding_count": 21},
        ),
        "MOTION": (
            {"effective_joint_track_count": 1},
            {"effective_joint_track_count": 0},
        ),
        "RUNTIME_CONSUMPTION": (
            {"representation_class": "DIRECTIONAL_2D_2P5D_PUPPET", "full_3d_reconstruction_authority": False},
            {"representation_class": "FULL_3D", "full_3d_reconstruction_authority": True},
        ),
    }
    assert all(mutation_worsens_measurement(domain, baseline, mutated) for domain, (baseline, mutated) in cases.items())


def test_production_lbs_kernel_is_not_imported_from_experiment_space():
    from pathlib import Path

    source = Path("compiler/realsas_compiler_core/deformation.py").read_text(encoding="utf-8")
    assert "experiments." not in source
    assert "realsas_compiler_services.numerics.lbs" in source


def test_canonical_compiler_core_has_no_experiment_runtime_imports():
    from pathlib import Path

    offenders = []
    for path in Path("compiler/realsas_compiler_core").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "from experiments" in source or "import experiments" in source:
            offenders.append(str(path))
    assert offenders == []
