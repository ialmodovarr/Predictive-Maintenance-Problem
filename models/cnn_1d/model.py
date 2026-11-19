"""Approved raw-acceleration/engineered-feature fusion network."""
import torch
from torch import nn

class HybridCNN(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.deep = nn.Sequential(
            nn.Conv1d(3,16,15,padding=7),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(16,32,7,padding=3),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.hidden=nn.Linear(77,32)
        self.dropout=nn.Dropout(dropout)
        self.output=nn.Linear(32,1)
        for layer in [*self.deep,self.hidden]:
            if isinstance(layer,(nn.Conv1d,nn.Linear)):
                nn.init.kaiming_normal_(layer.weight,nonlinearity="relu")
                nn.init.zeros_(layer.bias)
    def forward(self, acceleration, auxiliary):
        if acceleration.ndim!=3 or acceleration.shape[1]!=3 or auxiliary.shape!=(len(acceleration),13):
            raise ValueError("Expected acceleration (B,3,L), auxiliary (B,13)")
        fused=torch.cat((self.deep(acceleration),auxiliary),dim=1)
        return self.output(self.dropout(torch.relu(self.hidden(fused)))).squeeze(-1)
