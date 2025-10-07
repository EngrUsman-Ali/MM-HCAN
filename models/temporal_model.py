import torch
import torch.nn as nn

class TemporalFeatureExtractor(nn.Module):
    def __init__(self):
        super(TemporalFeatureExtractor, self).__init__()
        # 1D CNN Layers
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=512, batch_first=True)

    def forward(self, x):
        if x.dim() == 2:                       # add channel dim if missing
            x = x.unsqueeze(1)   
        x = self.cnn(x)  # (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        out, _ = self.lstm(x)
        return out[:, -1, :]  # Last time step -> (B, 512)