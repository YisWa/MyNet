CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
        --nproc_per_node=8  --master_port=1234 \
        --use_env main.py \
        --output_dir logs/r50_dec2_lr5e-5_refu \
        -c config/R50.py \
        --pretrained ./params/dino_r50.pth