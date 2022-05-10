import torch
from torch import nn, Tensor
from .backbones import HRNet

class PoseHRNet(nn.Module):
    def __init__(self, backbone: str = 'w32', num_joints: int = 17):
        # COCO Detection의 경우 17개의 조인트를 사용하고 있다.
        super().__init__()
        self.backbone = HRNet(backbone)
        self.final_layer = nn.Conv2d(self.backbone.all_channels[0], num_joints, 1)
        
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
    
    def forward(self, x: Tensor):
        out = self.backbone(x)
        out = self.final_layer(out)
        return out
    
if __name__ == '__main__':
    model = PoseHRNet('w48')
    x = torch.randn(1, 3, 256, 192)
    y = model(x)
    print(y.shape)