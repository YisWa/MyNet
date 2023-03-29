# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
import clip

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP

from .dn_components import prepare_for_cdn, dn_post_process


class DINO(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_verb_classes,
                 num_queries, aux_loss=False, interm_loss=False, clip_loss=False, num_feature_levels=1,
                 dec_pred_class_embed_share=True, dec_pred_bbox_embed_share=True,
                 two_stage_class_embed_share=True, two_stage_bbox_embed_share=True,
                 dn_number=100, dn_box_noise_scale=0.4, dn_label_noise_ratio=0.5):

        super().__init__()
        self.num_queries = num_queries
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.interm_loss = interm_loss
        self.clip_loss = clip_loss
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_verb_classes = num_verb_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.obj_label_enc = nn.Embedding(num_classes + 1, hidden_dim)
        self.verb_label_enc = nn.Linear(num_verb_classes, hidden_dim)

        # for dn training
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:  # NEED two_stage_type == 'no'
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        # prepare pred layers: class & box embed
        _class_embed = nn.Linear(hidden_dim, self.num_classes)
        _verb_class_embed = nn.Linear(hidden_dim, self.num_verb_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        _sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        _verb_class_embed.bias.data = torch.ones(self.num_verb_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(_sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_sub_bbox_embed.layers[-1].bias.data, 0)

        # init the detection head (for decoder output)
        if dec_pred_bbox_embed_share:  # True
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
            sub_box_embed_layerlist = [_sub_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
            sub_box_embed_layerlist = [copy.deepcopy(_sub_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:  # True
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
            verb_class_embed_layerlist = [_verb_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
            verb_class_embed_layerlist = [copy.deepcopy(_verb_class_embed) for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.sub_bbox_embed = nn.ModuleList(sub_box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.verb_class_embed = nn.ModuleList(verb_class_embed_layerlist)

        # refine point process
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.sub_bbox_embed = self.sub_bbox_embed

        # intermediate results detect head (for encoder output)
        if two_stage_bbox_embed_share:  # False
            assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
            self.transformer.enc_out_bbox_embed = _bbox_embed
            self.transformer.enc_out_sub_bbox_embed = _sub_bbox_embed
        else:
            self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            self.transformer.enc_out_sub_bbox_embed = copy.deepcopy(_sub_bbox_embed)

        if two_stage_class_embed_share:  # False
            assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
            self.transformer.enc_out_class_embed = _class_embed
            self.transformer.enc_out_verb_class_embed = _verb_class_embed
        else:
            self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
            self.transformer.enc_out_verb_class_embed = copy.deepcopy(_verb_class_embed)

        if self.clip_loss:
            self.sub_weight = nn.Linear(hidden_dim, 512)
            self.obj_weight = nn.Linear(hidden_dim, 512)
            self.hoi_weight = nn.Linear(hidden_dim, 512)
            self.fuse_norm = nn.LayerNorm(512)

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: NestedTensor, targets:List=None):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        # DeNosing Preprocess
        if self.dn_number > 0 or targets is not None:
            input_obj_labels, input_sub_labels, input_verb_labels, input_obj_boxes, input_sub_boxes, attn_mask, dn_meta = \
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale, samples.device),
                                training=self.training, num_queries=self.num_queries, num_classes=self.num_classes,
                                num_verb_classes=self.num_verb_classes, hidden_dim=self.hidden_dim,
                                obj_label_enc=self.obj_label_enc, verb_label_enc=self.verb_label_enc)
        else:
            assert targets is None
            input_obj_labels = input_sub_labels = input_verb_labels = None
            input_obj_boxes = input_sub_boxes = attn_mask = dn_meta = None

        # Transformer
        h_hs, h_ref, o_hs, o_ref, hoi, interm_class, interm_verb_class, interm_coord, interm_sub_coord = \
            self.transformer(srcs, masks, poss, input_obj_labels, input_sub_labels, input_verb_labels, input_obj_boxes, input_sub_boxes, attn_mask)
        # in case num_obj=0 or num_verb=0, so the gradient doesn't exist, which unable to use distributed training
        o_hs[0] += self.obj_label_enc.weight[0, 0] * 0.0
        hoi[0] += self.verb_label_enc.weight[0, 0] * 0.0 + self.verb_label_enc.bias[0] * 0.0

        # Detector head
        outputs_coord, outputs_sub_coord = [], []
        for l, (h_r, h_w, h_h, o_r, o_w, o_h) in enumerate(zip(h_ref[:-1], self.sub_bbox_embed, h_hs, o_ref[:-1], self.bbox_embed, o_hs)):
            sub_coord = (h_w(h_h) + inverse_sigmoid(h_r)).sigmoid()
            obj_coord = (o_w(o_h) + inverse_sigmoid(o_r)).sigmoid()
            outputs_sub_coord.append(sub_coord)
            outputs_coord.append(obj_coord)
        outputs_sub_coord, outputs_coord = torch.stack(outputs_sub_coord), torch.stack(outputs_coord)
        outputs_class = torch.stack([o_w(o_h) for o_w, o_h in zip(self.class_embed, o_hs)])
        outputs_verb_class = torch.stack([v_w(verb) for v_w, verb in zip(self.verb_class_embed, hoi)])

        # knowledge distillation
        if self.clip_loss:
            outputs_fuse_embed = torch.stack([self.fuse_norm(self.sub_weight(hum[:, -self.num_queries:, :]) +
                                                             self.obj_weight(obj[:, -self.num_queries:, :]) +
                                                             self.hoi_weight(verb[:, -self.num_queries:, :]))
                                              for hum, obj, verb in zip(h_hs, o_hs, hoi)])

        # DeNosing Postprocess
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_verb_class, outputs_coord, outputs_sub_coord = \
                dn_post_process(outputs_class, outputs_verb_class, outputs_coord, outputs_sub_coord,
                                dn_meta, self.aux_loss, self._set_aux_loss)

        # Generate results
        out = {'pred_obj_logits': outputs_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_coord[-1],
               'pred_feat': outputs_fuse_embed[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_verb_class, outputs_coord, outputs_sub_coord)
            for i in range(len(outputs_fuse_embed) - 1):
                out['aux_outputs'][i]['pred_feat'] = outputs_fuse_embed[i]
        if self.interm_loss:  # for encoder output
            out['interm_outputs'] = {'pred_obj_logits': interm_class, 'pred_verb_logits': interm_verb_class,
                                     'pred_sub_boxes': interm_sub_coord, 'pred_obj_boxes': interm_coord}
        out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]


class SetCriterion(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, focal_alpha, focal_gamma, losses):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.losses = losses

        self.clip_model, _ = clip.load('ViT-B/32', device="cuda")

    def loss_mimic(self, outputs, targets):
        src_feats = torch.mean(outputs['pred_feat'], dim=1)
        img_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            img_feats = self.clip_model.encode_image(img_inputs)

        loss_feat_mimic = F.l1_loss(src_feats, img_feats)
        losses = {'loss_feat': loss_feat_mimic}
        return losses

    def loss_obj_labels(self, outputs, targets, indices, num_interactions):
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_obj_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_interactions, alpha=self.focal_alpha,
                                         gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_obj_ce': loss_obj_ce}

        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes
        }
        return loss_map[loss](outputs, targets, indices, num)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        dn_meta = outputs['dn_meta']
        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['obj_labels']) > 0:
                    t = torch.arange(0, len(targets[i]['obj_labels'])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            l_dict = {}
            for loss in self.losses:
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_interactions * scalar))
            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][i]
                    l_dict = {}
                    for loss in self.losses:
                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_interactions * scalar))
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_interactions)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # knowledge distillation
        if 'pred_feat' in outputs:
            l_dict = self.loss_mimic(outputs, targets)
            losses.update(l_dict)
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                l_dict = self.loss_mimic(aux_outputs, targets)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size=dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad=pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups


class PostProcess(nn.Module):

    def __init__(self, subject_category_id, nms_number):
        super().__init__()
        self.subject_category_id = subject_category_id
        self.nms_number = nms_number

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = \
            outputs['pred_obj_logits'], outputs['pred_verb_logits'], outputs['pred_sub_boxes'], outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        verb_scores = out_verb_logits.sigmoid()

        # NMS
        num_verb_classes = verb_scores.shape[-1]
        num_obj_classes = obj_prob.shape[-1]

        topk_values, topk_indexes = torch.topk(obj_prob.flatten(1), self.nms_number, dim=1)
        obj_scores = topk_values
        topk_boxes = topk_indexes // num_obj_classes
        obj_labels = topk_indexes % num_obj_classes

        # top 100
        verb_scores = torch.gather(verb_scores, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, num_verb_classes))
        out_obj_boxes = torch.gather(out_obj_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        out_sub_boxes = torch.gather(out_sub_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)
            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})

        return results


def build_dino(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_deformable_transformer(args)

    model = DINO(
        backbone,
        transformer,
        num_classes=args.num_obj_classes,
        num_verb_classes=args.num_verb_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        interm_loss=args.interm_loss,
        clip_loss=args.clip_loss,
        num_feature_levels=args.num_feature_levels,
        dec_pred_class_embed_share=args.dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=args.dec_pred_bbox_embed_share,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        dn_number=args.dn_number if args.use_dn else 0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
    )
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {}
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    # for DN training
    if args.use_dn:
        weight_dict['loss_obj_ce_dn'] = args.obj_loss_coef
        weight_dict['loss_verb_ce_dn'] = args.verb_loss_coef
        weight_dict['loss_sub_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_obj_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_sub_giou_dn'] = args.giou_loss_coef
        weight_dict['loss_obj_giou_dn'] = args.giou_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.interm_loss:
        interm_weight_dict = {}
        interm_weight_dict.update({k + f'_interm': v * args.interm_loss_coef for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    # knowledge distillation
    if args.clip_loss:
        mimic_weight_dict = {'loss_feat': args.clip_loss_coef}
        for i in range(args.dec_layers - 1):
            mimic_weight_dict.update({k + f'_{i}': v for k, v in mimic_weight_dict.items()})
        weight_dict.update(mimic_weight_dict)

    losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes']
    criterion = SetCriterion(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                             weight_dict=weight_dict, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
                             losses=losses)
    criterion.to(device)
    postprocessors = {'hoi': PostProcess(args.subject_category_id, args.nms_number)}

    return model, criterion, postprocessors
