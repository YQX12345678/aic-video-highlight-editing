import torch
from src.model import VideoCropModel
from src.loss import temporal_iou_loss

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for frames, targets in dataloader:
        frames, targets = frames.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(frames)
        l1 = torch.nn.functional.l1_loss(preds, targets)
        iou = temporal_iou_loss(preds, targets)
        loss = l1 + iou
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
