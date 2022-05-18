import torch
import math
from torch import nn, Tensor

class HeatmapGEval(nn.Module):
    def __init__(self, sigma=1):
        super().__init__()
        self.sigma = sigma
    def __call__(self, pred:Tensor, answer:Tensor):
        
        B_pred, C_pred, H_pred, W_pred = pred.shape
        B_ans, C_ans, H_ans, W_ans = answer.shape
        
        assert B_pred == B_ans, 'Batch size Error, Check the codes'
        assert C_pred == C_ans, 'Channel size Error, Check the codes'
        assert H_pred == H_ans, 'Height size Error, Check the codes'
        assert W_pred == W_ans, 'Width size Error, Check the codes'
        b = int(B_pred) # b as batch_size
        o = torch.zeros(size=(B_pred, ))
        for bidx in range(b):
            pred_sampled, ans_sampled = pred[bidx].flatten(), answer[bidx].flatten()
            pred_sampled -= ans_sampled
            sum = 0.0

            for c in pred_sampled:
                sum += (1/(2 * math.pi * self.sigma)) * math.exp(-1 * ((c**2) / (2 * self.sigma ** 2)))

            o[bidx] = (2 * math.pi * sum) / (C_pred * H_pred * W_pred)
        
        return o
            
            
if __name__ == '__main__':
    from copy import deepcopy
    
    testclass = HeatmapGEval(sigma=1)
    pred_rand = torch.randn((1, 29, 64, 64))
    answ_rand = deepcopy(pred_rand)
    o = testclass(pred_rand, answ_rand)
    
    for oo in o:
        print(oo)