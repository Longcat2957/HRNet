# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
import torch
from torch import nn, Tensor

class JointsMSELoss(nn.Module):
    # heatmap 기반 loss
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')           # loss function 으로 평균 제곱 오차 함수를 사용
        self.use_target_weight = use_target_weight
        
    def forward(self, output:Tensor, target:Tensor, target_weight):                 # output sample(minibatch = 16) -> torch.Size([16, 29, 2])
        batch_size = output.size(0)                                                 # same as out_channel
        num_joints = output.size(1)                                                 # same as joints_number
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)    # reshape to tuple
                                                                                    # (16, 29, 2) -> (29, 16, 2)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1,1)       # same shape as heatmaps_pred
        loss = 0                                                                    # loss variable initialize
        
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()                             # len(heatmap_pred) = batch_size
            heatmap_gt = heatmaps_gt[idx].squeeze()                                 # len(heatmap_gt) = batch_size
            if self.use_target_weight:                                              # if use_target_weight is exsist
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        
        return loss / num_joints

class JointsOHKMMSELoss(nn.Module):
    # skeleton 기반 loss
    def __init__(self, use_target_weight, topk=8):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        
    def ohkm(self, loss:Tensor):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                # Returns the k largest elements of the given input tensor
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss
    
    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)