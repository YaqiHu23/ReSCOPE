import torch.nn as nn


class ROIScorer(nn.Module):
    def __init__(self, in_ch, roi_tiles, stride_tiles, hidden=256):
        super().__init__()
        self.roi_tiles = int(roi_tiles)
        self.stride_tiles = int(stride_tiles)

        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.head = nn.Conv2d(
            hidden,
            1,
            kernel_size=self.roi_tiles,
            stride=self.stride_tiles,
            padding=0,
        )

    def forward(self, x):
        return self.head(self.backbone(x))
