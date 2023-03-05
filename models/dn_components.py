# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F


def prepare_for_cdn(dn_args, training, num_queries, num_classes, num_verb_classes, hidden_dim, obj_label_enc, verb_label_enc):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale, device = dn_args
        # positive and negative dn queries
        dn_number = dn_number * 2  # pos/neg number -> pos+neg number
        known = [(torch.ones_like(t['obj_labels'])).to(device) for t in targets]  # [1 1] [1 1 1]
        batch_size = len(known)  # 2
        known_num = [sum(k) for k in known]  # [2 3]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))  # pos/neg number per group
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        obj_labels = torch.cat([t['obj_labels'] for t in targets])  # (n,)
        verb_labels = torch.cat([t['verb_labels'] for t in targets])  # (n, 117)
        obj_boxes = torch.cat([t['obj_boxes'] for t in targets])  # (n, 4)
        sub_boxes = torch.cat([t['sub_boxes'] for t in targets])  # (n, 4)
        batch_idx = torch.cat([torch.full_like(t['obj_labels'].long(), i) for i, t in enumerate(targets)])  # [0 0 1 1 1]

        known_obj_labels = obj_labels.repeat(2 * dn_number, 1).view(-1)  # (n*dn_number*2,)
        known_verb_labels = verb_labels.repeat(2 * dn_number, 1)  # (n*dn_number*2, 117)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)  # [0 0 1 1 1]*dn_number*2
        known_obj_boxes = obj_boxes.repeat(2 * dn_number, 1)  # (n*dn_number*2, 4)
        known_sub_boxes = sub_boxes.repeat(2 * dn_number, 1)  # (n*dn_number*2, 4)
        known_obj_labels_expand, known_verb_labels_expand = known_obj_labels.clone(), known_verb_labels.clone()
        known_obj_boxes_expand, known_sub_boxes_expand = known_obj_boxes.clone(), known_sub_boxes.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_obj_labels_expand.float())
            chosen_indices = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob

            new_label = torch.randint_like(chosen_indices, 0, num_classes)  # randomly put a new one here
            known_obj_labels_expand.scatter_(0, chosen_indices, new_label)

            new_index = torch.randint_like(chosen_indices, 0, num_verb_classes)  # for 117
            replace_index = new_label * num_verb_classes + new_index
            known_verb_labels_expand = known_verb_labels_expand.view(-1)
            known_verb_labels_expand.scatter_(0, replace_index, torch.ones(len(chosen_indices)).to(device))
            known_verb_labels_expand = known_verb_labels_expand.view(-1, num_verb_classes)

        single_pad = int(max(known_num))
        pad_size = int(single_pad * 2 * dn_number)
        
        positive_idx = torch.tensor(range(len(obj_boxes))).long().to(device).unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(obj_boxes) * 2).long().to(device).unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(obj_boxes)
        
        def gen_boxes(known_boxes, known_boxes_expand):
            
            known_bbox_ = torch.zeros_like(known_boxes)
            known_bbox_[:, :2] = known_boxes[:, :2] - known_boxes[:, 2:] / 2
            known_bbox_[:, 2:] = known_boxes[:, :2] + known_boxes[:, 2:] / 2

            diff = torch.zeros_like(known_boxes)
            diff[:, :2] = known_boxes[:, 2:] / 2
            diff[:, 2:] = known_boxes[:, 2:] / 2

            rand_sign = torch.randint_like(known_boxes, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_boxes)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).to(device) * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_boxes_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_boxes_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

            return known_boxes_expand

        if box_noise_scale > 0:
            known_sub_boxes_expand = gen_boxes(known_sub_boxes, known_sub_boxes_expand)
            known_obj_boxes_expand = gen_boxes(known_obj_boxes, known_obj_boxes_expand)

        input_obj_labels_embed = obj_label_enc(known_obj_labels_expand.long().to(device))
        input_verb_labels_embed = verb_label_enc(known_verb_labels_expand.float().to(device))
        input_obj_boxes_embed = inverse_sigmoid(known_obj_boxes_expand)
        input_sub_boxes_embed = inverse_sigmoid(known_sub_boxes_expand)

        input_obj_labels = torch.zeros(batch_size, pad_size, hidden_dim).to(device)
        input_sub_labels = torch.randn(batch_size, pad_size, hidden_dim).to(device)  # TODO how distribution zeros ones or randn
        input_verb_labels = torch.zeros(batch_size, pad_size, hidden_dim).to(device)
        input_obj_boxes = torch.zeros(batch_size, pad_size, 4).to(device)
        input_sub_boxes = torch.zeros(batch_size, pad_size, 4).to(device)

        map_known_indice = torch.tensor([]).to(device)
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_obj_labels[(known_bid.long(), map_known_indice)] = input_obj_labels_embed
            input_verb_labels[(known_bid.long(), map_known_indice)] = input_verb_labels_embed
            input_obj_boxes[(known_bid.long(), map_known_indice)] = input_obj_boxes_embed
            input_sub_boxes[(known_bid.long(), map_known_indice)] = input_sub_boxes_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {'pad_size': pad_size, 'num_dn_group': dn_number}

    else:
        input_obj_labels = input_sub_labels = input_verb_labels = None
        input_obj_boxes = input_sub_boxes = None
        attn_mask = dn_meta = None

    return input_obj_labels, input_sub_labels, input_verb_labels, input_obj_boxes, input_sub_boxes, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_verb_class, outputs_coord, outputs_sub_coord, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_verb_class = outputs_verb_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        output_known_sub_coord = outputs_sub_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_verb_class = outputs_verb_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        outputs_sub_coord = outputs_sub_coord[:, :, dn_meta['pad_size']:, :]
        out = {'pred_obj_logits': output_known_class[-1], 'pred_verb_logits': output_known_verb_class[-1],
               'pred_sub_boxes': output_known_sub_coord[-1], 'pred_obj_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_verb_class,
                                               output_known_sub_coord, output_known_coord)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_verb_class, outputs_coord, outputs_sub_coord


