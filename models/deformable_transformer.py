# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
from typing import Optional

import torch
from torch import nn, Tensor

from util.misc import inverse_sigmoid
from .utils import gen_encoder_output_proposals, MLP,_get_activation_fn, gen_sineembed_for_position
from .ops.modules import MSDeformAttn
from util.box_ops import box_union, compute_spatial_feature


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_queries=300, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.0, activation="relu", query_dim=4, num_feature_levels=1,
                 enc_n_points=4, dec_n_points=4):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, d_model, query_dim)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))  # for lvl_pos_embed
        self.tgt_embed = nn.Embedding(self.num_queries, d_model)  # decoder embedding
        nn.init.normal_(self.tgt_embed.weight.data)
        self.sub_tgt_embed = nn.Embedding(self.num_queries, d_model)  # decoder embedding
        nn.init.normal_(self.sub_tgt_embed.weight.data)
        self.hoi_tgt_embed = nn.Embedding(self.num_queries, d_model)  # decoder embedding
        nn.init.normal_(self.hoi_tgt_embed.weight.data)

        # anchor selection at the output of encoder
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None
        self.enc_out_sub_class_embed = None
        self.enc_out_sub_bbox_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, poss, tgt, sub_tgt, hoi_tgt, refpoint_embed, sub_refpoint_embed, attn_mask=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer
            
        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, poss)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # Encoder
        memory = self.encoder(src=src_flatten, pos=lvl_pos_embed_flatten, level_start_index=level_start_index,
                              spatial_shapes=spatial_shapes, valid_ratios=valid_ratios, key_padding_mask=mask_flatten)

        # initial Static_Content_Queries and Dynamic_Anchors
        output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)  # (bs, sum(hi*wi), d_model) (bs, sum(hi*wi), 4)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))  # (bs, sum(hi*wi), d_model)

        # for select topK proposals
        enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)  # (bs, sum(hi*wi), 80)
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], self.num_queries, dim=1)[1]  # bs, nq

        # for reduce calculation, select first and then calculate
        memory_proposals = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))  # (bs, nq, d_model)
        box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # (bs, nq, 4) unsigmoid

        # gather boxes: Dynamic_Anchors
        refpoint_embed_undetach = self.enc_out_bbox_embed(memory_proposals) + box_proposal  # (bs, nq, 4) unsigmoid
        refpoint_embed_ = refpoint_embed_undetach.detach()  # (bs, nq, 4) unsigmoid
        sub_refpoint_embed_undetach = self.enc_out_sub_bbox_embed(memory_proposals) + box_proposal  # (bs, nq, 4) unsigmoid
        sub_refpoint_embed_ = sub_refpoint_embed_undetach.detach()  # (bs, nq, 4) unsigmoid

        interm_class = self.enc_out_class_embed(memory_proposals)  # (bs, nq, num_obj_classes)
        interm_verb_class = self.enc_out_verb_class_embed(memory_proposals)  # (bs, nq, num_verb_classes)

        # gather tgt: Static_Content_Queries
        tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # bs, nq, d_model
        sub_tgt_ = self.sub_tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # TODO for human init content
        hoi_tgt_ = self.hoi_tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # TODO for hoi init content

        if refpoint_embed is not None:  # for DN
            refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)  # (bs, dn+nq, 4) unsigmoid
            tgt = torch.cat([tgt, tgt_], dim=1)  # (bs, dn+nq, d_model)
            sub_refpoint_embed = torch.cat([sub_refpoint_embed, sub_refpoint_embed_], dim=1)  # TODO for concat human bbox unsigmoid
            sub_tgt = torch.cat([sub_tgt, sub_tgt_], dim=1)  # TODO for concat human init content
            hoi_tgt = torch.cat([hoi_tgt, hoi_tgt_], dim=1)  # TODO for concat hoi init content
        else:
            refpoint_embed, tgt = refpoint_embed_, tgt_
            sub_refpoint_embed, sub_tgt, hoi_tgt = sub_refpoint_embed_, sub_tgt_, hoi_tgt_  # TODO for concat human & hoi init content

        # Decoder
        h_hs, h_ref, o_hs, o_ref, hoi = self.decoder(  # (n_dec, bs, nq, d_model)    (n_dec+1, bs, bq, 4) sigmoid
                tgt=tgt.transpose(0, 1),  # dn+nq, bs, d_model
                sub_tgt=sub_tgt.transpose(0, 1),  # TODO for human content
                hoi=hoi_tgt.transpose(0, 1),  # TODO for hoi content
                memory=memory.transpose(0, 1),  # sum(hi*wi), bs, d_model
                memory_key_padding_mask=mask_flatten,  # bs, sum(hi*wi)  MASK
                refpoints_unsigmoid=refpoint_embed.transpose(0, 1),  # dn+nq, bs, 4
                sub_refpoints_unsigmoid=sub_refpoint_embed.transpose(0, 1),  # TODO for human bbox
                level_start_index=level_start_index,  # num_levels
                spatial_shapes=spatial_shapes,  # num_levels, 2
                valid_ratios=valid_ratios,  # bs, num_levels, 2
                tgt_mask=attn_mask)  # (dn+nq, dn+nq)

        return h_hs, h_ref, o_hs, o_ref, hoi, interm_class, interm_verb_class, \
               refpoint_embed_undetach.sigmoid(), sub_refpoint_embed_undetach.sigmoid()


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, enc_layer_share=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, pos, spatial_shapes, level_start_index, valid_ratios, key_padding_mask):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus: 
            - output: [bs, sum(hi*wi), 256]
        """
        output = src
        # preparation and reshape
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)  # (bs, sum(hi*wi), level, 2)

        # main process
        for layer_id, layer in enumerate(self.layers):
            output = layer(src=output, pos=pos, reference_points=reference_points, spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index, key_padding_mask=key_padding_mask)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, d_model=256, query_dim=4, dec_layer_share=False):
        super().__init__()

        self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        self.sub_layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        self.hoi_layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)

        self.norm, self.sub_norm, self.hoi_norm = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.sub_ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        # self.hoi_ref_point_head = MLP(2 * d_model, d_model, d_model, 2)

        fuser = Fuser(d_model, activation="relu", d_ffn=d_model*8, dropout=0.0)
        self.fusers = _get_clones(fuser, num_layers, layer_share=False)

        self.bbox_embed = None
        self.sub_bbox_embed = None

    def forward(self, tgt, sub_tgt, hoi, memory,  # 1100, bs, d_model   # sum(hi*wi), bs, d_model
                tgt_mask: Optional[Tensor] = None,  # dn+nq, dn+nq
                memory_key_padding_mask: Optional[Tensor] = None,  # bs, sum(hi*wi)
                refpoints_unsigmoid: Optional[Tensor] = None, # 1100, bs, 4
                sub_refpoints_unsigmoid: Optional[Tensor] = None,  # 1100, bs, 4
                # for memory
                level_start_index: Optional[Tensor] = None,  # num_levels
                spatial_shapes: Optional[Tensor] = None,  # num_levels, 2
                valid_ratios: Optional[Tensor] = None,  # bs, num_levels, 2
                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output, sub_output, hoi_output = tgt, sub_tgt, hoi

        intermediate, sub_intermediate, hoi_intermediate = [], [], []
        reference_points, sub_reference_points = refpoints_unsigmoid.sigmoid(), sub_refpoints_unsigmoid.sigmoid()  # dn+nq, bs, 4
        ref_points, sub_ref_points = [reference_points], [sub_reference_points]

        for layer_id, (layer, sub_layer, hoi_layer, fuser) in enumerate(zip(self.layers, self.sub_layers, self.hoi_layers, self.fusers)):

            reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # dn+nq, bs, num_levels, 4
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # dn+nq, bs, d_model*2
            query_pos = self.ref_point_head(query_sine_embed)  # dn+nq, bs, d_model

            sub_reference_points_input = sub_reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # dn+nq, bs, num_levels, 4
            sub_query_sine_embed = gen_sineembed_for_position(sub_reference_points_input[:, :, 0, :]) # dn+nq, bs, d_model*2
            sub_query_pos = self.sub_ref_point_head(sub_query_sine_embed)  # dn+nq, bs, d_model

            fused_pos, sub_fused_pos = fuser.pair_fusion(output, query_pos, sub_output, sub_query_pos)

            # object block
            # (dn+nq,bs,d_model) (dn+nq,bs,d_model) (dn+nq,bs,num_levels,4) (sum(hi*wi),bs,d_model) (bs,sum(hi*wi))
            # (num_levels,) (num_levels, 2) (dn+nq,dn+nq) => (dn+nq,bs,d_model)
            output = layer(output, fused_pos, reference_points_input, memory, memory_key_padding_mask,
                           level_start_index, spatial_shapes, tgt_mask)
            new_reference_points = (self.bbox_embed[layer_id](output) + inverse_sigmoid(reference_points)).sigmoid()  # dn+nq, bs, 4

            # human block
            sub_output = sub_layer(sub_output, sub_fused_pos, sub_reference_points_input, memory, memory_key_padding_mask,
                                   level_start_index, spatial_shapes, tgt_mask)
            sub_new_reference_points = (self.sub_bbox_embed[layer_id](sub_output) + inverse_sigmoid(sub_reference_points)).sigmoid()  # dn+nq, bs, 4

            # interaction block
            hoi_reference_points = box_union(new_reference_points, sub_new_reference_points)
            hoi_reference_points_input = hoi_reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]

            # reference_points_input = new_reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            # query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # dn+nq, bs, d_model*2
            # sub_reference_points_input = sub_new_reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # dn+nq, bs, num_levels, 4
            # sub_query_sine_embed = gen_sineembed_for_position(sub_reference_points_input[:, :, 0, :])  # dn+nq, bs, d_model*2
            # hoi_query_pos = self.hoi_ref_point_head(torch.cat([output, sub_output], dim=-1))  # dn+nq, bs, d_model

            hoi_fused_pos = fuser.ternary_fusion(output, new_reference_points, sub_output, sub_new_reference_points)

            hoi_output = hoi_layer(hoi_output, hoi_fused_pos, hoi_reference_points_input, memory, memory_key_padding_mask,  # TODO use hoi_fused_output or hoi_query_pos
                                   level_start_index, spatial_shapes, tgt_mask)
            # hoi_output = fuser.ternary_fusion(sub_output, output, hoi_output, hoi_query_pos)
            # hoi_output = fuser.ffn(hoi_output)

            reference_points, sub_reference_points = new_reference_points.detach(), sub_new_reference_points.detach()  # dn+nq, bs, 4

            ref_points.append(new_reference_points)  # 7, dn+nq, bs, 4
            intermediate.append(self.norm(output))  # 6, dn+nq, bs, d_model
            sub_ref_points.append(sub_new_reference_points)  # 7, dn+nq, bs, 4
            sub_intermediate.append(self.sub_norm(sub_output))  # 6, dn+nq, bs, d_model
            hoi_intermediate.append(self.hoi_norm(hoi_output))  # 6, dn+nq, bs, d_model

        return [
            [sub_itm_out.transpose(0, 1) for sub_itm_out in sub_intermediate],
            [sub_itm_refpoint.transpose(0, 1) for sub_itm_refpoint in sub_ref_points],
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
            [hoi_itm_out.transpose(0, 1) for hoi_itm_out in hoi_intermediate],
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):

        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_sa(self, tgt, tgt_query_pos, self_attn_mask):
        # self attention
        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt

    def forward_ca(self, tgt, tgt_query_pos, tgt_reference_points, memory, memory_key_padding_mask,
                   memory_level_start_index, memory_spatial_shapes):
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1), tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward_ffn(self, tgt):
        # feed forward network
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, tgt_query_pos, tgt_reference_points, memory, memory_key_padding_mask,
                memory_level_start_index, memory_spatial_shapes, self_attn_mask):
        # self-attn
        tgt = self.forward_sa(tgt, tgt_query_pos, self_attn_mask)
        # cross-attn
        tgt = self.forward_ca(tgt, tgt_query_pos, tgt_reference_points, memory, memory_key_padding_mask,
                              memory_level_start_index, memory_spatial_shapes)
        # FFN
        tgt = self.forward_ffn(tgt)

        return tgt


class Fuser(nn.Module):
    def __init__(self, d_model, activation, d_ffn, dropout):
        super().__init__()

        # pair_fusion
        self.sub_weight1 = nn.Linear(d_model, d_model)
        self.obj_weight1 = nn.Linear(d_model, d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_act = _get_activation_fn(activation)
        self.sub_weight2 = nn.Linear(d_model, d_model)
        self.obj_weight2 = nn.Linear(d_model, d_model)
        self.sub_norm = nn.LayerNorm(d_model)
        self.obj_norm = nn.LayerNorm(d_model)

        # ternary_fusion
        self.sub_weight3 = nn.Linear(d_model, d_model)
        self.sub_norm = nn.LayerNorm(d_model)
        self.sub_act = _get_activation_fn(activation)
        self.sub_weight4 = nn.Linear(d_model, d_model)

        self.obj_weight3 = nn.Linear(d_model, d_model)
        self.obj_norm = nn.LayerNorm(d_model)
        self.obj_act = _get_activation_fn(activation)
        self.obj_weight4 = nn.Linear(d_model, d_model)

        self.hoi_weight1 = nn.Linear(16, 64)
        self.hoi_norm1 = nn.LayerNorm(64)
        self.hoi_act = _get_activation_fn(activation)
        self.hoi_weight2 = nn.Linear(64, d_model)
        self.hoi_norm2 = nn.LayerNorm(d_model)
        #
        #
        # # ffn
        # self.linear1 = nn.Linear(d_model, d_ffn)
        # self.activation = _get_activation_fn(activation)
        # self.dropout1 = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(d_ffn, d_model)
        # self.dropout2 = nn.Dropout(dropout)
        # self.ffn_norm = nn.LayerNorm(d_model)

    def pair_fusion(self, sub_content, sub_position, obj_content, obj_position):
        # Xc = ReLU(LN(Ws0 * Xs + Wo0 * Xo))  Xs' = Xs + Ws1 * Xc + Ps  Xo' = Xo + Wo1 * Xc + Po
        cross_content = self.cross_act(self.cross_norm(self.sub_weight1(sub_content) + self.obj_weight1(obj_content)))
        new_sub_position = self.sub_norm(self.sub_weight2(cross_content) + sub_position)
        new_obj_position = self.obj_norm(self.obj_weight2(cross_content) + obj_position)
        return new_sub_position, new_obj_position

    def ternary_fusion(self, obj_content, obj_position, sub_content, sub_position):
        # Xc = Ws2 * ReLU(LN(Ws1 * Xs)) +Wo2 * ReLU(LN(Wo1 * Xo))  Xh' = LN(Xh + Xc + Wc * ReLU(Ws3 * Ps + Wo3 * Po))
        cross_content = self.sub_weight4(self.sub_act(self.sub_norm(self.sub_weight3(sub_content)))) + \
                        self.obj_weight4(self.obj_act(self.obj_norm(self.obj_weight3(obj_content))))
        spatial_feature = compute_spatial_feature(sub_position, obj_position)
        cross_position = self.hoi_weight2(self.hoi_act(self.hoi_norm1(self.hoi_weight1(
            torch.cat([sub_position, obj_position, spatial_feature], dim=-1)))))
        new_hoi_position = self.hoi_norm2(cross_content + cross_position)
        return new_hoi_position
    #
    # def ffn(self, tgt):
    #     # feed forward network
    #     tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
    #     tgt = tgt + self.dropout2(tgt2)
    #     tgt = self.ffn_norm(tgt)
    #     return tgt


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_deformable_transformer(args):

    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
    )
