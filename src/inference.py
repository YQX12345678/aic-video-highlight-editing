import json
import torch
from src.model import VideoCropModel

def generate_submission(model_path, video_ids, output_file, device='cuda', seq_len=30):
    model = VideoCropModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    with torch.no_grad():
        for vid in video_ids:
            dummy_frames = torch.randn(1, seq_len, 3, 224, 224).to(device)
            preds = model(dummy_frames).squeeze(0).cpu().numpy()
            video_dict = {
                "video_id": str(vid),
                "targetRatioWH": [16, 9],
                "predictions": []
            }
            for frame_idx, box in enumerate(preds):
                video_dict["predictions"].append({
                    "frame": frame_idx,
                    "bboxes": [float(b) for b in box]
                })
            results.append(video_dict)

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '
')
    print(f'结果已保存至 {output_file}')
