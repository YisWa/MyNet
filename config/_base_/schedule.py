# Learning Rate
lr = 5e-5
param_dict_type = 'default'
lr_backbone = 5e-6
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
lr_drop = 50
lr_drop_list = [50, 80]

# Train
seed = 42
batch_size = 2
weight_decay = 0.0001
epochs = 60
start_epoch = 0
clip_max_norm = 0.1
save_checkpoint_interval = 10
eval_interval = 1
print_frequency = 10

# Dist
device = 'cuda'
world_size = 1
dist_url = 'env://'
rank = 0
find_unused_params = True
