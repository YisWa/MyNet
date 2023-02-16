# Backbone
dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None

# Transformer
enc_layers = 6
dec_layers = 6
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True
transformer_activation = 'relu'

# Matcher
set_cost_obj_class = 1.0
set_cost_verb_class = 1.0
set_cost_bbox = 2.5
set_cost_giou = 1.0

# Loss
aux_loss = True
obj_loss_coef = 1.0
verb_loss_coef = 1.0
bbox_loss_coef = 2.5
giou_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25
focal_gamma = 2.0
verb_loss_type = 'focal'

# DeNoising
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
