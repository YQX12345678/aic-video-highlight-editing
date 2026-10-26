"""Video data preprocessing and frame sampling."""
import cv2

def sample_frames(video_path: str, target_fps: int = 1) -> list:
    """Uniformly sample frames from a video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(fps / target_fps))
    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames
