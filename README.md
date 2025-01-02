# skip norm

Implementing classification ViT at scale with deepspeed acceleration. Investigating the effect of normalizing the residual stream on training stability
- Weights and Biases: https://wandb.ai/sugolov/skipnorm/

## Setup
Dependencies
```
pip install torch==2.3.1 torchvision==0.18.1 deepspeed==0.16.2
``` 

### Run single device training (`run.sh`):
```
python run.py \
    --data-path ~/.data \
    --num-workers 1
    --epochs 1 \
    --checkpoint-path checkpoints \
    --checkpoint-epochs 1
```
### Run distributed training (`run_deepspeed.sh`)
```
deepspeed \
    --num_gpus=2 \
    run.py \
    --deepspeed \
    --deepspeed_config ds_configs/default.json \
    --data-path /fs01/datasets/cifar10 \
    --epochs 1 \
    --num-workers 2 \
    --checkpoint-path /fs01/projects/skipnorm/checkpoints \
    --checkpoint-epoch 1
```

