"""Post-processing for video highlight/cropping trajectories."""

def smooth_bboxes(predictions: list, window_size: int = 5) -> list:
    """Apply moving average to bounding boxes to reduce temporal jitter."""
    if not predictions: return []
    smoothed = []
    for i in range(len(predictions)):
        start = max(0, i - window_size // 2)
        end = min(len(predictions), i + window_size // 2 + 1)
        window = [p["bboxes"] for p in predictions[start:end]]
        avg_bbox = [sum(w[j] for w in window) / len(window) for j in range(len(predictions[i]["bboxes"]))]
        smoothed.append({"frame": predictions[i]["frame"], "bboxes": avg_bbox})
    return smoothed
