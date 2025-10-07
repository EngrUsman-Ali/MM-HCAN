import torch
import torch.nn as nn
import torch.nn.functional as F

class HGNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HGNNLayer, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, L):
        x = F.relu(self.conv1(L @ x))
        x = self.conv2(L @ x)
        return x

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0).mean()
        return loss