module purge
module load cuda-11.3
module load nccl_2.9.9-1+cuda11.3
module list

export PATH=/pkgs/anaconda3/bin:$PATH
export PYTHONPATH=$HOME/condaenvs/pytorch-1.3:$PYTHONPATH
export LD_LIBRARY_PATH=/pkgs/cuda-10.0/lib64:/pkgs/cudnn-10.0-v7.6.3.30/lib64:$LD_LIBRARY_PATH

source /fs01/projects/skipnorm/.env/torch-env/bin/activate
which python

deepspeed \
    --num_gpus=2 \
    run.py \
    --deepspeed \
    --deepspeed_config /fs01/projects/skipnorm/skipnorm_source/ds_configs/default.json \
    --data-path /fs01/datasets/cifar10 \
    --epochs 1 \
    --num-workers 2 \
    --checkpoint-path /fs01/projects/skipnorm/checkpoints \
    --checkpoint-epoch 1