import torch

def temporal_iou_loss(pred_boxes, gt_boxes, epsilon=1e-6):
    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    gt_x1 = gt_boxes[..., 0] - gt_boxes[..., 2] / 2
    gt_x2 = gt_boxes[..., 0] + gt_boxes[..., 2] / 2

    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter = torch.clamp(inter_x2 - inter_x1, min=0)

    union = (pred_x2 - pred_x1) + (gt_x2 - gt_x1) - inter
    iou = inter / (union + epsilon)
    return 1.0 - iou.mean()
