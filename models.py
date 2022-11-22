import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUModel(nn.Module):
    def __init__(self, num_class, num_feature):
        super().__init__()
        backbone = nn.GRU(num_feature, 64, 4, batch_first=True).cuda()
        self.norm = nn.LayerNorm(normalized_shape=[100,6])
        self.backbone = backbone
        self.num_class = num_class
        self.head = nn.Sequential(
            nn.LazyLinear(1024),nn.ReLU(),
            nn.LazyLinear(num_class))

    def forward(self, X):
        # X = self.norm(X)
        self.h = torch.randn(4, X.shape[0], 64).cuda()
        y, self.h = self.backbone(X, self.h)
        y = self.head(y.reshape(y.shape[0], -1))
        y = F.softmax(y, dim=1)
        return y