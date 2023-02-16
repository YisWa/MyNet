# Learning Rate
lr = 1e-4
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
lr_drop = 120
lr_drop_list = [90, 130]

# Train
seed = 42
batch_size = 2
weight_decay = 0.0001
epochs = 150
start_epoch = 0
clip_max_norm = 0.1
save_checkpoint_interval = 10
eval_interval = 10

# EMA
use_ema = False
ema_decay = 0.9997
ema_epoch = 0

# Dist
device = 'cuda'
world_size = 1
dist_url = 'env://'
rank = 0
amp = False  # mixed precision
find_unused_params = False
