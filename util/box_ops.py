# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch, math
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)


    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)



# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2) # N, 4

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def box_union(boxes1, boxes2):
    """
    计算两组框的并集
    Args:
        boxes1: 第一组框，形状为[bs, n, 4]
        boxes2: 第二组框，形状为[bs, n, 4]
    Returns:
        并集框，形状为[bs, n, 4]
    """
    # 获取框的左上角和右下角坐标
    boxes1_left_top = boxes1[..., :2] - boxes1[..., 2:] / 2
    boxes1_right_bottom = boxes1[..., :2] + boxes1[..., 2:] / 2
    boxes2_left_top = boxes2[..., :2] - boxes2[..., 2:] / 2
    boxes2_right_bottom = boxes2[..., :2] + boxes2[..., 2:] / 2

    # 计算并集框的左上角和右下角坐标
    union_left_top = torch.min(boxes1_left_top, boxes2_left_top)
    union_right_bottom = torch.max(boxes1_right_bottom, boxes2_right_bottom)

    # 计算并集框的宽和高
    union_w = union_right_bottom[..., 0] - union_left_top[..., 0]
    union_h = union_right_bottom[..., 1] - union_left_top[..., 1]

    # 计算并集框的中心点横坐标、纵坐标、宽度和高度
    union_x = (union_left_top[..., 0] + union_right_bottom[..., 0]) / 2
    union_y = (union_left_top[..., 1] + union_right_bottom[..., 1]) / 2
    union_w = torch.clamp(union_w, min=0)  # 宽度可能为负数，需要裁剪
    union_h = torch.clamp(union_h, min=0)  # 高度可能为负数，需要裁剪

    # 组合并集框的坐标和尺寸
    union_boxes = torch.stack([union_x, union_y, union_w, union_h], dim=-1)

    return union_boxes


def compute_spatial_feature(boxes1, boxes2):

    dx = boxes1[:, :, 0] - boxes2[:, :, 0]
    dy = boxes1[:, :, 1] - boxes2[:, :, 1]
    dist = torch.stack([dx, dy], dim=-1)
    dist = torch.cat([dist, torch.norm(dist, dim=-1, keepdim=True)], dim=-1)
    dist = torch.cat([dist, (torch.atan2(dy, dx) / math.pi).unsqueeze(-1)], dim=-1)

    area1 = (boxes1[:, :, 2] * boxes1[:, :, 3]).unsqueeze(-1)
    area2 = (boxes2[:, :, 2] * boxes2[:, :, 3]).unsqueeze(-1)

    boxes1_lt, boxes1_rb = boxes1[..., :2] - boxes1[..., 2:] / 2, boxes1[..., :2] + boxes1[..., 2:] / 2
    boxes2_lt, boxes2_rb = boxes2[..., :2] - boxes2[..., 2:] / 2, boxes2[..., :2] + boxes2[..., 2:] / 2

    inter_lt = torch.max(boxes1_lt, boxes2_lt)
    inter_rb = torch.min(boxes1_rb, boxes2_rb)
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = (inter_wh[..., 0] * inter_wh[..., 1]).unsqueeze(-1)
    union = area1 + area2 - inter
    return torch.cat([dist, area1, area2, inter, union], dim=-1)


if __name__ == '__main__':
    x = torch.tensor([[[3,2,2,2],[3,2,2,2]],[[3,2,2,2],[3,2,2,2]]]).float()
    y = torch.tensor([[[2,3,2,2],[2,3,2,2]],[[2,3,2,2],[2,3,2,2]]]).float()
    b = compute_spatial_feature(x, y)
    print(b)