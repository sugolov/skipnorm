import sys
sys.path.append("../traj-norm")
import torch

from skipnorm import SkipNorm

def test_initialization():
    sn = SkipNorm(normalized_shape=10, window_size=5, elementwise_affine=True)
    print(sn.weight)

def test_mlp():
    from sn_models import SNResNet
    mlp = SNResNet(5, dim=5, hidden_dim=5, sn_depth=2)
    x = torch.randn((5,))
    print(mlp(x))


if __name__ == "__main__":
    # test_initialization()
    test_mlp()
