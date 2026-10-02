import pytest

torch=pytest.importorskip('torch')
from models.arachne.v6.readout_v6 import ArachneV6RawReadout


def test_readout_chunking_and_row_joint_permutations_preserve_predictions():
    torch.manual_seed(17)
    model=ArachneV6RawReadout(8,6,10,relation_dim=8).eval()
    memory=torch.randn(7,8);geom=torch.randn(7,7);pair=torch.randn(7,3,10);tokens=torch.randn(3,4,6)
    legal=torch.ones(7,3,dtype=torch.bool);legal[:,1]=False
    with torch.inference_mode():
        _,a=model.decode_all(memory,geom,pair,tokens,legal,chunk=2)
        _,b=model.decode_all(memory,geom,pair,tokens,legal,chunk=7)
        rows=torch.tensor([4,0,6,2,1,5,3]);joints=torch.tensor([2,0,1])
        _,p=model.decode_all(memory[rows],geom[rows],pair[rows][:,joints],tokens[joints],legal[rows][:,joints],chunk=3)
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(a[rows][:,joints],p)
    torch.testing.assert_close(a.sum(-1),torch.ones(7))
    assert torch.count_nonzero(a[:,1])==0
