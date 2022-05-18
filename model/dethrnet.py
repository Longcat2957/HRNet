import math
import torch
import numpy as np
from torch import nn, Tensor
from backbones import HRNet


class DetrHRNet(nn.Module):
    """
    DETR(Detection with Transformer)
    Attribute
    =========
    bacbone: str = 'w18', 'w32', 'w48'

    """
    def __init__(self, backbone: str = 'w32'):
        super().__init__()
        # Feature extracting
        self.backbone = HRNet(backbone)
        self.conv = nn.Conv2d(in_channels=32, out_channels=256, kernel_size=1) # hidden_dim as 512
        # Feature extracting
        
        
        # DETR architecture
        self.transformer = nn.Transformer(256, 8, 6, 6)
        self.linear_class = nn.Linear(256, 4) # num_class = 1 # class # 0 -> N/A and class # 1 -> Human
        self.linear_bbox = nn.Linear(256, 4)
        
        self.query_pos = nn.Parameter(torch.rand(128, 256))
        self.row_embed = nn.Parameter(torch.rand(64, 256//2))
        self.col_embed = nn.Parameter(torch.rand(64, 256//2))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
        # 초기 weight를 입력하는 부분이다.
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def init_pretrained(self, pretrained:str = None):
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
    
    def forward(self, inputs: Tensor):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        return self.linear_class(h), self.linear_bbox(h).sigmoid()
    
if __name__ == '__main__':
    model = DetrHRNet()
    x = torch.randn(1, 3, 256, 256)
    l, b = model(x)
    print(l.shape, b.shape)
    # l -> logits, b -> bboxes
    probas = l.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7