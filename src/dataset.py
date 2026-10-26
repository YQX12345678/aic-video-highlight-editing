import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os

class VideoCropDataset(Dataset):
    def __init__(self, video_dir, annotation_df, transform=None, seq_len=30):
        self.video_dir = video_dir
        self.annotations = annotation_df
        self.transform = transform
        self.seq_len = seq_len

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_path = os.path.join(self.video_dir, str(row['video_id']) + '.mp4')
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, row['start_frame'])
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()

        while len(frames) < self.seq_len:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        frames = np.stack(frames)
        frames = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1) / 255.0
        target_boxes = torch.tensor(row['target_boxes'], dtype=torch.float32) 
        return frames, target_boxes
