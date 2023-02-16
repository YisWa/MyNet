_base_ = ['./_base_/dataset.py',
          './_base_/model.py',
          './_base_/schedule.py']

backbone = 'swin_L_384_22k'
backbone_dir = './params'
use_checkpoint = True
