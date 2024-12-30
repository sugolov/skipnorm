# skip norm

To run training:
```
python train.py --cifar
``` 

## TODO
- get a ViT to train >90% accuracy on cifar10, use DEiT data augmentation + warmup strategies
- implement attention skip
- if it seems to "work", implement across a few models
 
```
python -m pip install torch deepspeed torchvision wandb 
```