import torch
import torch.nn as nn


class LogitNormHead(nn.Module):

    def __init__(self, num_classes, num_verb_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_verb_classes = num_verb_classes
        self.register_buffer("obj_moving_mean", torch.zeros(self.num_classes))
        self.register_buffer("obj_moving_var", torch.ones(self.num_classes))
        self.register_buffer("verb_moving_mean", torch.zeros(self.num_verb_classes))
        self.register_buffer("verb_moving_var", torch.ones(self.num_verb_classes))
        self.momentum = 5e-4
        self.eps = 1e-5

    def get_statistics(self):
        obj_mean_val = self.obj_moving_mean
        obj_std_val = torch.sqrt(torch.clamp(self.obj_moving_var, min=1e-11))
        obj_beta = obj_mean_val.min()

        verb_mean_val = self.verb_moving_mean
        verb_std_val = torch.sqrt(torch.clamp(self.verb_moving_var, min=1e-11))
        verb_beta = verb_mean_val.min()

        return obj_mean_val, obj_std_val, obj_beta, verb_mean_val, verb_std_val, verb_beta

    def gather(self, holder, input, reduction='ma'):
        with torch.no_grad():
            if reduction == 'ma':
                holder = (1.0 - self.momentum) * holder + self.momentum * input
            elif reduction == 'sum':
                holder += input
        return holder

    def forward(self, pred_obj_logits, pred_verb_logits):

        if self.training:
            # basic statistics
            self.obj_moving_mean = self.gather(self.obj_moving_mean, pred_obj_logits.flatten(0, 1).mean(0))
            self.obj_moving_var = self.gather(self.obj_moving_var, pred_obj_logits.flatten(0, 1).var(0))
            self.verb_moving_mean = self.gather(self.verb_moving_mean, pred_verb_logits.flatten(0, 1).mean(0))
            self.verb_moving_var = self.gather(self.verb_moving_var, pred_verb_logits.flatten(0, 1).var(0))

        else:
            obj_mean_val, obj_std_val, obj_beta, verb_mean_val, verb_std_val, verb_beta = self.get_statistics()
            pred_obj_logits = (pred_obj_logits - (obj_mean_val - obj_beta)) / obj_std_val
            pred_verb_logits = (pred_verb_logits - (verb_mean_val - verb_beta)) / verb_std_val

        return pred_obj_logits, pred_verb_logits


def build_head(args):
    return LogitNormHead(num_classes=args.num_obj_classes, num_verb_classes=args.num_verb_classes)

# if __name__ == '__main__':
#     a = torch.arange(1, 25).view(2,3,4).float()
#     b = torch.arange(25, 49).view(2,3,4).float()
#     model = LogitNormHead(4,4)
#     model.train()
#     out = model(a,b)
#     print(out)
#     print(model.obj_moving_mean, model.obj_moving_var)
#     print(model.verb_moving_mean, model.verb_moving_var)
#
#     print("-"*20)
#     model.eval()
#     out = model(a, b)
#     print(out)
