TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
        --nproc_per_node=8  --master_port=1234 \
        --use_env main.py \
        --output_dir logs/swinl_test \
        -c config/SwinL.py \
        --pretrained ./params/dino_swinl.pth
