CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
        --nproc_per_node=8  --master_port=1234 \
        --use_env main.py \
        --output_dir logs/swinT_copy \
        -c config/DINO/DINO_4scale_swin.py \
        --options dn_scalar=100 dn_label_coef=1.0 dn_bbox_coef=1.0 backbone_dir=/data1/public_dataset/params \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det