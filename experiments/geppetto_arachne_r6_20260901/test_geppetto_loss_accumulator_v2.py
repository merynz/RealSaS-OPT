import torch

from experiments.geppetto_arachne_r6_20260901.geppetto_loss_v2 import _independent_zero_accumulators


def test_geppetto_loss_accumulators_do_not_alias_storage():
    zero = torch.tensor(0.0, requires_grad=True) * 1.0
    names = ("position_nll", "existence", "stop", "root", "parent", "support", "support_presence", "abstain")
    acc = _independent_zero_accumulators(zero, names)
    ptrs = [acc[name].data_ptr() for name in names]
    assert len(set(ptrs)) == len(ptrs)
    acc["position_nll"] += 3.0
    assert float(acc["position_nll"].detach()) == 3.0
    assert all(float(acc[name].detach()) == 0.0 for name in names if name != "position_nll")
