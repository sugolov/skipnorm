cd skipnorm_source

/fs01/projects/skipnorm/.env/bin/python -m deepspeed \
    --num_gpus=4 \
    run.py \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --data-path /fs01/datasets/cifar10 \
    --epochs 1 \
    --checkpoint-path /fs01/projects/skipnorm/checkpoints \
    --checkpoint-epochs 1