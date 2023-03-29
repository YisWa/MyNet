_base_ = ['./_base_/dataset.py',
          './_base_/model.py',
          './_base_/schedule.py']

backbone = 'swin_L_384_22k'
backbone_dir = '/data1/public_dataset/params'
use_checkpoint = True
