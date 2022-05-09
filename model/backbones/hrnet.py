import torch
from torch import nn, Tensor
"""
import Pytorch
import torch.nn, torch.Tensor for nn.Module, nn.Sequenital container inheritance
"""

class Conv(nn.Sequential):
    """
    Pytorch Conv2D의 경우 입력층과 출력층의 사이즈를 자동으로 계산한다.
    
    Affine fuction으로 Convolution2D && BatchNormalization 실행
    Activation fuction으로 ReLU를 사용한다.
    """
    def __init__(self, inchannel, outchannel, kernel_size, strides=1, padding=0):
        super().__init__(
            nn.Conv2d(inchannel, outchannel, kernel_size, strides, padding, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(True)
        )
        

class BasicBlock(nn.Module):
    """
    * Attribute
    ===========
    self.conv1, self.bn1, self.relu
    self.conv2, self.bn2
    self.downsample, self.stride
    
    * Method
    ========
    self.forward(x:Tensor) -> Tensor
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # inplanes = in_channels, planes = out_channels
        super().__init__()                                                          # in_Channel, Out_Channel, Kernel_size, stride, padding 
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)          # N,        , N          , 3          , 1     , zeropadding = 1
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)
        
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)                 # N,        , N          , 3          , 1     , zeropadding = 1
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: Tensor):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)                                           # x1 -> x2 -> x4 와 같은 다운샘플링을 수행하기도 한다.
        
        out += identity                                                             # Idea of Residual Network
        out = self.relu(out)                                                        # Activation function == ReLu
        
        return out
    
class Bottleneck(nn.Module):
    """
    * Attribute
    ===========
    self.conv1, self.bn1
    self.conv2, self.bn2
    self.conv3, self.bn3
    self.relu
    self.downsample
    self.stride
    
    * Method
    ========
    self.forward(x:Tensor) -> Tensor
    """
    expansion = 4 # Constants
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)                     # kernel_size =1
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)            # kernel_size = 3, padding = 1, stride = 1 
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, 1, bias=False)        # Out_Channel의 숫자가 In_Channel에 비해 4배 증가했으므로
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)                            # (x, y, C) -> (0.5x, 0.5y, 4C) 형태로 변환된다.
        
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x:Tensor):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out
    
class HRModule(nn.Module):
    """
    * Attribute
    ===========
    self.num_branches
    self.branches
    self.fuse_layers
    self.relu
    
    
    * Private Method
    ================
    self._make_fuse_layers(num_branches, num_channels, ms_output)
    
    * Method
    ========
    self.forward(x:Tensor) -> Tensor
    
    """
    def __init__(self, num_branches, num_channels, ms_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.brances = nn.ModuleList([
            nn.Sequential(*[
                BasicBlock(num_channels[i], num_channels[i])
            for _ in range(4)])
        for i in range(num_branches)])
        
        self.fuse_layers = self._make_fuse_layers(num_branches, num_channels, ms_output)
        self.relu = nn.ReLU(True)
        
    def _make_fuse_layers(self, num_branches, num_channels, ms_output=True):
        fuse_layers = []
        
        for i in range(num_branches if ms_output else 1):
            fuse_layer = []
            
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                            nn.BatchNorm2d(num_channels[i]),
                            nn.Upsample(scale_factor=2**(j-1), mode='nearest')
                        )
                    )
                elif j==i:
                    fuse_layer.append(None)
                else: # j < i, need Downsample by 3x3kernel
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_channels[j], num_channels[i], 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_channels[i])
                                )
                            )
                        else:
                            conv3x3s.append(Conv(num_channels[j], num_channels[j], 3, 2, 1))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
            
        return nn.ModuleList(fuse_layers)
    
    def forward(self, x:Tensor):
        for i, m in enumerate(self.branches):
            x[i] = m(x[i])

        x_fuse = []
        
        for i, fm in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fm[0](x[0])
            
            for j in range(1, self.num_branches):
                y = y + x[j] if i == j else y + fm[j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse
    

# depth에 따른 해상도 차이
hrnet_settings ={
    "w18": [18, 36, 72, 144],
    "w32": [32, 64, 128, 256],
    "w48": [48, 96, 192, 384]
}

class HRNet(nn.Module):
    """
    * Attributes
    ============
    self.conv1, self.bn1, self.conv2, self.bn2, self.relu
    self.all_channels
    self.layer1
    self.transition1, self.stage2
    self.transition2, self.stage3
    self.transition3, self.stage4
    
    
    * Private Method
    ================
    self._make_layer(inplanes, planes, blocks) -> nn.Sequential(*layers)
    >> 일반적인(regular) conv 레이어를 생성하는 내부 메서드
        
    self._make_transition_layer(c1s, c2s) -> nn.ModuleList(transition_layers)
    >> transition conv 레이어를 생성하는 내부 메서드
    >> trans conv. 를 통해 low res의 feature를 high res에 propagation
        
    self._make_stage(num_modules, num_branches, num_channels, ms_output) -> nn.Sequential(*modules)
    >>
    
    * Method
    ========
    self.forward(x: Tensor) -> Tensor

    """
    def __init__(self, backbone: str = 'w18'):
        super().__init__()
        assert backbone in hrnet_settings.keys(), f"HRNet model name should be in {list(hrnet_settings.keys())}"

        
        # stem
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)              # RGB 3 Channels to 64 Channels
        self.bn1 = nn.BatchNorm2d(64)  
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)             # 64 Channels to 64 Channels
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        
        self.all_channels = hrnet_settings[backbone]
        # if self.all_channels = [18, 36, 72, 144]        


        # Stage 1
        self.layer1 = self._make_layer(64, 64, 4)                                                   # In_Channel 64, Out_Channel 64, Blocks # = 4; 레이어 통과 전후 크기 변화 없음 && 4개의 블록 생성
        stage1_out_channel = Bottleneck.expansion * 64                                              # Bottleneck.expansion = 4 (상수) 이므로 stage_out_channel = 256 
        
        # Stage 2
        stage2_channels = self.all_channels[:2]                                                     # [18, 36]
        self.transition1 = self._make_transition_layer([stage1_out_channel], stage2_channels)       # self._make_transition_layer([256], [18, 36])
        self.stage2 = self._make_stage(1, 2, stage2_channels)                                       # 
        
        # Stage 3
        stage3_channels = self.all_channels[:3]                                                     # [18, 36, 72]
        self.transition2 = self._make_transition_layer(stage2_channels, stage3_channels)
        self.stage3 = self._make_stage(4, 3, stage3_channels)
        
        # Stage 4
        self.transition3 = self._make_transition_layer(stage3_channels, self.all_channels)
        self.stage4 = self._make_stage(3, 4, self.all_channels, ms_output=False)                    # [18, 36, 72, 144]
    
    def _make_layer(self, inplanes, planes, blocks):
        """
        paramters
        =========
        inplanes : 입력층 채널 수
        planes : 출력 층 채널 수
        blocks : 반복할 횟수
        """
        downsample = None
        if inplanes != planes * Bottleneck.expansion: # Bottleneck.expansion (Constant) = 4
            # downsampling을 수행하는 AN을 생성한다.
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*Bottleneck.expansion, 1, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion)
            )
        
        layers = []
        layers.append(Bottleneck(inplanes, planes, downsample=downsample))
        inplanes = planes * Bottleneck.expansion
        
        for _ in range(1, blocks):
            layers.append(Bottleneck(inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def _make_transition_layer(self, c1s, c2s):
        """
        parameters
        ==========
        c1s : 
        c2s : 
        
        """
        num_branches_pre = len(c1s)
        num_branches_cur = len(c2s)
        
        transition_layers = []
        
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if c1s[i] != c2s[i]:
                    transition_layers.append(Conv(c1s[i], c2s[i], 3, 1, 1))
                else:
                    transition_layers.append(None)
                    
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = c1s[-1]
                    outchannels = c2s[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(Conv(inchannels, outchannels, 3, 2, 1))
                transition_layers.append(nn.Sequential(*conv3x3s))
        
        return nn.ModuleList(transition_layers) # 서로 다른 해상도의 (x1, x2, x4, x8) 서로 병렬 구조의 레이어 리스트를 리턴한다.
    
    def _make_stage(self, num_modules, num_branches, num_channels, ms_output=True):
        """
        parameters
        ==========
        num_modules
        num_branches
        num_channels
        ms_output
        
        """
        modules = []

        for i in range(num_modules):
            # multi-scale output is only used in last module
            if not ms_output and i == num_modules - 1:
                reset_ms_output = False
            else:
                reset_ms_output = True
            modules.append(HRModule(num_branches, num_channels, reset_ms_output))

        return nn.Sequential(*modules)
    
    def forward(self, x:Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Stage 1
        x = self.layer1(x)

        # Stage 2
        x_list = [trans(x) if trans is not None else x for trans in self.transition1]
        y_list = self.stage2(x_list)

        # Stage 3
        x_list = [trans(y_list[-1]) if trans is not None else y_list[i] for i, trans in enumerate(self.transition2)]
        y_list = self.stage3(x_list)

        # # Stage 4
        x_list = [trans(y_list[-1]) if trans is not None else y_list[i] for i, trans in enumerate(self.transition3)]
        y_list = self.stage4(x_list)
        return y_list[0]
    
if __name__ == '__main__':
    model = HRNet('w32')
    print('model initialized')
