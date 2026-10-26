"""Evaluation metrics for video bbox tracking."""

def compute_bbox_iou(pred_box: list, gt_box: list) -> float:
    """Compute 1D IoU for vertical bounds [x, y1, y2]."""
    if len(pred_box) == 3 and len(gt_box) == 3:
        inter = max(0, min(pred_box[2], gt_box[2]) - max(pred_box[1], gt_box[1]))
        union = max(pred_box[2], gt_box[2]) - min(pred_box[1], gt_box[1])
        return inter / union if union > 0 else 0.0
    return 0.0
