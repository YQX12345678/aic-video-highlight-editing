import torch
import torch.nn as nn
import torchvision.models as models

class VideoCropModel(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_layers=3):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_in = nn.Linear(512, feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Linear(feature_dim, 4)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.cnn(x).view(B, T, -1) 
        features = self.fc_in(features)
        temporal_features = self.transformer(features) 
        boxes = self.regressor(temporal_features) 
        return torch.sigmoid(boxes)
