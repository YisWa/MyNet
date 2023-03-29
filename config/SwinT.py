_base_ = ['./_base_/dataset.py',
          './_base_/model.py',
          './_base_/schedule.py']

backbone = 'swin_T_224_1k'
backbone_dir = '/data1/public_dataset/params'
use_checkpoint = True
