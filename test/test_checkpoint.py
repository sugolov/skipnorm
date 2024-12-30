import os
import torch

def test_load_cpt(loc="checkpoints/ViT_T16_1.pt"):
    print(torch.load(loc).keys())

if __name__ == "__main__":
    test_load_cpt()