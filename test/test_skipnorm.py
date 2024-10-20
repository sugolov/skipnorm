import sys
sys.path.append("../skipnorm")
import torch

from skipnorm import SkipNorm

def test_initialization():
    sn = SkipNorm(normalized_shape=10, window_size=5, elementwise_affine=True)
    print(sn.weight)

def test_mlp():
    from sn_models import SNResNet
    mlp = SNResNet(5, dim=5, hidden_dim=5, sn_window=2)
    x = torch.randn((5,))
    print(mlp(x))

def test_transformer():
    from sn_models import SNTransformer
    transformer = SNTransformer(dim=16, depth=4, heads=2, dim_head=16, mlp_dim=16, window=2)
    x = torch.randn((1, 4, 16,))
    print(transformer(x))


if __name__ == "__main__":
    # test_initialization()
    # test_mlp()
    test_transformer()
