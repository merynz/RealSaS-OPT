"""Verify gradients, not scientific outcomes; initialization has an isolated RNG."""
import runpy
from pathlib import Path
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
joint = runpy.run_path(str(HERE / "continuous_joint_system_probe_v2.py"))
for arm in ("S2_JOINT_BOUND", "FULL_SHARED_JOINT_BOUND", "S2_JOINT_UNBOUND", "S2_JOINT_BOUND_RANDOM_RELATION"):
    torch.manual_seed(611)
    model = joint["CourtModel"](arm, "DENSE3D", 2)
    a = torch.rand(2, 4, 3) * 1.4 - 0.7
    b = torch.rand(2, 4, 3) * 1.4 - 0.7
    logits = model.pair(a, b)["relation_logits"]
    F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
    for name, parameter in model.named_parameters():
        relation_private = name.startswith(("relation.", "global_gp."))
        if relation_private:
            assert parameter.grad is not None, (arm, name)
        else:
            assert parameter.grad is None, (arm, name)
    print("RELATION_GRADIENT_BOUNDARY_PASS", arm, flush=True)
endpoint = runpy.run_path(str(HERE / "observation_endpoint_interaction_probe_v2.py"))
for private, bound in ((True, True), (False, True), (True, False)):
    torch.manual_seed(611)
    model = endpoint["CourtModel"](private_subspaces=private, bound_relation=bound)
    batch = endpoint["make_batch"](2, generator=torch.Generator().manual_seed(511), alpha_mode="raw", random_relation=False)
    F.binary_cross_entropy_with_logits(model(batch)["relation"], batch["relation_target"]).backward()
    for name, parameter in model.named_parameters():
        if name.startswith("relation."):
            assert parameter.grad is not None, name
        else:
            assert parameter.grad is None, name
    print("ENDPOINT_GRADIENT_BOUNDARY_PASS", private, bound, flush=True)
