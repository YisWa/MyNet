import clip, torch, math
import torch.nn as nn
from datasets.data_stats import hico_hoi_text_label, hico_obj_text_label
from datasets.data_stats import vcoco_obj_text_label, vcoco_hoi_text_label


class CLIP(nn.Module):
    def __init__(self, hidden_dim, clip_embed_dim, num_obj_classes, num_verb_classes):
        super().__init__()
        self.hoi_logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        if self.args.dataset_file == 'hico':
            hoi_text_label = hico_hoi_text_label
            obj_text_label = hico_obj_text_label
        elif self.args.dataset_file == 'vcoco':
            hoi_text_label = vcoco_hoi_text_label
            obj_text_label = vcoco_obj_text_label
        hoi_clip_label, obj_clip_label = self.init_classifier_with_clip(hoi_text_label, obj_text_label)

        self.hoi_class_fc = nn.Sequential(nn.Linear(hidden_dim, clip_embed_dim), nn.LayerNorm(clip_embed_dim))
        self.hoi_visual_projection = nn.Linear(clip_embed_dim, num_verb_classes)
        self.hoi_visual_projection.weight.data = hoi_clip_label / hoi_clip_label.norm(dim=-1, keepdim=True)

        self.obj_class_fc = nn.Sequential(nn.Linear(hidden_dim, clip_embed_dim), nn.LayerNorm(clip_embed_dim))
        self.obj_visual_projection = nn.Linear(clip_embed_dim, num_obj_classes)
        self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)

    def _reset_parameters(self):
        pass

    def init_classifier_with_clip(self, hoi_text_label, obj_text_label):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hoi_text_inputs = torch.cat([clip.tokenize(hoi_text[1]) for hoi_text in hoi_text_label])
        obj_text_inputs = torch.cat([clip.tokenize(obj_text[1]) for obj_text in obj_text_label])
        clip_model, _ = clip.load(self.args.clip_model, device=device)
        with torch.no_grad():
            hoi_text_embedding = clip_model.encode_text(hoi_text_inputs.to(device))
            obj_text_embedding = clip_model.encode_text(obj_text_inputs.to(device))

        del clip_model

        return hoi_text_embedding.float(), obj_text_embedding.float()

    def forward(self, o_hs, inter_hs):
        obj_logit_scale = self.obj_logit_scale.exp()
        o_hs = self.obj_class_fc(o_hs)
        o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
        outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)

        hoi_logit_scale = self.hoi_logit_scale.exp()
        inter_hs = self.hoi_class_fc(inter_hs)
        inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
        outputs_hoi_class = hoi_logit_scale * self.hoi_visual_projection(inter_hs)

        return outputs_obj_class, outputs_hoi_class


def build_clip(args):
    return CLIP(hidden_dim=args.hidden_dim, clip_embed_dim=512, num_obj_classes=args.num_obj_classes,
                num_verb_classes=args.num_verb_classes)
