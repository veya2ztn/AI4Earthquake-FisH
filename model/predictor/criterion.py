import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from einops.layers.torch import Rearrange

from torch.distributions import Normal

class GrowthImportanceMSE(_Loss):
    __constants__ = ['reduction']

    def __init__(self, weight, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight',weight) 
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape)==1:target = target.unsqueeze(-1)
        return torch.mean(self.weight * (input - target) ** 2) 
class GrowthImportanceABS(_Loss):
    __constants__ = ['reduction']

    def __init__(self, weight, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight',weight) 
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape)==1:target = target.unsqueeze(-1)
        return torch.mean(self.weight * (input - target).abs()) 

class BroadCastMSE(nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor, weight:Optional[Tensor]=None) -> Tensor:
        reduction = 'mean'
        """
        a^2*(x-y)^2 and a >=0 ==> (ax-ay)^2
        """
        # if len(target.shape)==2 and len(input.shape)==3 and input.shape[1]==1:
        #     input = input.squeeze(1)
        assert len(input.shape) == len(target.shape), f"the input shape is {input.shape} but the target shape is {target.shape}"
        if weight is not None:
            assert len(weight.shape) == len(target.shape), f"the weight shape is {weight.shape} but the target shape is {target.shape}"
            weight = weight/(weight.sum(-1,keepdim=True) + 1e-5) # <-- before code is wrong
            weight = torch.sqrt(weight)
            assert not torch.isnan(weight).any()
            input  = input * weight
            target = target * weight
            error = F.mse_loss(input, target, reduction='none')
            error = error.sum(1).mean(0)
            return error
        else:
            if input.shape[-1] != target.shape[-1] and target.shape[-1] == 1:
                target = target.expand_as(input)
            return F.mse_loss(input, target, reduction=reduction)
    
class FastABS(nn.MSELoss):
    __constants__ = ['reduction']

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape)==1:target = target.unsqueeze(-1)
        if len(input.shape) == len(target.shape):
            input = input.squeeze(-1)
        return torch.mean((input - target).abs()) 

class CosineSimilirity(nn.Module):
    __constants__ = ['reduction']

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert target.shape[-1]==2, f" the target must be a 2D vector, but the shape is {target.shape} "
        assert  input.shape[-1]==2, f" the target must be a 2D vector, but the shape is {input.shape} "
        if len(input.shape) == len(target.shape) + 1:
            target = target.unsqueeze(1) # (B,2) -> (B,1,2)
        if len(target.shape) == len(input.shape) + 1:
            input = input.unsqueeze(1) # (B,2) -> (B,1,2)
        return 1-torch.nn.functional.cosine_similarity(input, target, dim=-1).mean()

class ParallelQSimilirity(nn.Module):
    __constants__ = ['reduction']

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert target.shape[-1]==2, f" the target must be a 2D vector, but the shape is {target.shape} "
        assert  input.shape[-1]==2, f" the target must be a 2D vector, but the shape is {input.shape} "
        if len(input.shape) == len(target.shape) + 1:
            target = target.unsqueeze(1) # (B,2) -> (B,1,2)
        if len(target.shape) == len(input.shape) + 1:
            input = input.unsqueeze(1) # (B,2) -> (B,1,2)
        return 1 - (torch.nn.functional.cosine_similarity(input, target, dim=-1)**2).mean()



class VectorDistance(nn.Module):
    __constants__ = ['reduction']

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert target.shape[-1]==2, f" the target must be a 2D vector, but the shape is {target.shape} "
        assert  input.shape[-1]==2, f" the target must be a 2D vector, but the shape is {input.shape} "
        if len(input.shape) == len(target.shape) + 1:
            target = target.unsqueeze(1) # (B,2) -> (B,1,2)
        if len(target.shape) == len(input.shape) + 1:
            input = input.unsqueeze(1) # (B,2) -> (B,1,2)
        return (target - input).norm(dim=-1).mean()


class CosineLoss(nn.Module):
    __constants__ = ['reduction']

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape)==1:target = target.unsqueeze(-1)
        input = input.squeeze(-1)
        return torch.mean((torch.cos(input*np.pi) -  torch.cos(target*np.pi))**2 + (torch.sin(input*np.pi) -  torch.sin(target*np.pi))**2)

class AdaptCrossEntropyLoss(nn.Module):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.permute(0, 2 , 1)
        return F.cross_entropy(input, target.long())




class KLDivPosition(nn.Module):
    '''
         input: (B, L) is a distribution
        target: (B,  ) is a number in [0,1] indicate the label position in [0, L]
    '''

    std = 0.02
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        L = input.shape[1]
        target = target.squeeze(-1)
        assert len(target.shape)==1, f"the target shape is {target.shape}"
        dist = Normal(target, self.std)
        step_size = 1/L
        x = torch.linspace(0, 1 + step_size, L+1) - step_size / 2
        cdf_at_bin_boundaries = dist.cdf(x.unsqueeze(-1).to(input.device))
        pmf = cdf_at_bin_boundaries[1:] - cdf_at_bin_boundaries[:-1]
        pmf = pmf.T
        input = torch.log_softmax(input, dim=1)
        return torch.nn.functional.kl_div(input, pmf)*L #<--- kl div accept log input and normal target

class DistributionPosition(nn.Module):
    '''
         input: (B, L) is a distribution
        target: (B,  ) is a number in [0,1] indicate the label position in [0, L]
    '''
    def __init__(self, std:float=0.02, judger_alpha:float=0):
        super().__init__()
        self.std = std
        self.judger_alpha = judger_alpha
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ## input  (B, N, L+1) or (B, L+1)
        ## target (B, N) or (B, )
        #print(input.shape, target.shape)
        if len(input.shape) == 3:
            input  = input.flatten(0,1)  #(BN,L+1)
            target = target.flatten(0,1) #(BN )
        
        L = input.shape[1] - 1 # Notice the first dimension is used for judge
        target = target.squeeze(-1)
        assert len(target.shape)==1, f"the target shape is {target.shape} and input shape is {input.shape}"
        dist = Normal(target, self.std)
        step_size = 1/L
        x = torch.linspace(0, 1 + step_size, L+1) - step_size / 2
        cdf_at_bin_boundaries = dist.cdf(x.unsqueeze(-1).to(input.device))
        pmf = cdf_at_bin_boundaries[1:] - cdf_at_bin_boundaries[:-1]
        pmf = pmf.T
        residuel = 1 - pmf.sum(1,keepdim=True)
        pmf      = torch.cat([residuel, pmf], 1)
        input    = torch.softmax(input, dim=1)
        if self.judger_alpha == 0:
            return torch.sum((input -  pmf)**2) 
        else:
            weight = torch.tensor([self.judger_alpha] + [(1-self.judger_alpha)/L]*(L), device=input.device).unsqueeze(0)
            return torch.sum((input -  pmf)**2 * weight) #(B, L+1) * (1, L+1) -> (B, L+1) -> (B, ) -> scalar
            

from torchvision.ops.focal_loss import sigmoid_focal_loss
class FocalLoss(nn.Module):
    '''
         input: (B, L) is a distribution
        target: (B,  ) is a number in [0,1] indicate the label position in [0, L]
    '''
    def __init__(self, alpha:float=0.25, gamma:float=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.squeeze(-1)
        target= target.squeeze(-1)
        assert input.shape == target.shape, f"the input shape is {input.shape} but the target shape is {target.shape}"
        return sigmoid_focal_loss(input,target,alpha=self.alpha, gamma=self.gamma).mean()
    



class AntiLoss(nn.Module):
    """
    The input and target should be both in [-1,1 ]
    
    loss =  1 - (1 - |e|)**2
    """
    __constants__ = ['reduction']

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape)==1:target = target.unsqueeze(-1)
        input = input.squeeze(-1)
        return 1 - torch.mean(((input - target).abs() - 1 )**2)



class UncertaintyLoss(nn.Module):
    def __init__(self, shape, weight=1, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        self.max_logvar    = nn.Parameter(torch.randn(1,*shape))
        self.min_logvar    = nn.Parameter(0.1*torch.randn(1,*shape)) 
        self.weight        = weight
    
    def forward(self, preded: Tensor, target: Tensor) -> Tensor:
        ## assume the target is fixed sequence like (B,1) or (B,L)
        ## assume the input must be fixed sequence like (B,L,2)
        if len(target.shape)==1:target = target.unsqueeze(-1)
        predict_mean, log_var = preded.chunk(2, dim=-1)
        predict_mean = predict_mean.squeeze(-1)
        log_var = log_var.squeeze(-1)

        log_var = self.max_logvar - F.softplus(self.max_logvar - log_var)
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)
        predict_std = torch.exp(log_var/2)

        normal_func = torch.distributions.Normal(predict_mean, predict_std)
        log_prob    = normal_func.log_prob(target)
        reward = 0 - self.weight * (predict_mean.float().detach() - target.float().detach()).abs()
        loss = torch.mean(reward * log_prob) + 0.01 * torch.mean(self.max_logvar) - 0.01 * torch.mean(self.min_logvar)
        return loss

class Possloss(nn.Module):
    
    def __init__(self, shape, weight=1, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        self.max_logvar    = nn.Parameter(torch.randn(1,*shape))
        self.min_logvar    = nn.Parameter(0.1*torch.randn(1,*shape)) 
        self.weight        = weight
        self.inc_var_loss  = True

    def forward(self, pred:torch.Tensor, target:torch.Tensor):
        # print(pred.shape, target.shape, self.max_logvar.shape, self.min_logvar.shape)
        inc_var_loss = self.inc_var_loss # kwargs.get("inc_var_loss", True)
        loss_weight  = 1
        weight  = self.weight
        
        if loss_weight is None:
            loss_weight = 1        
        num_examples  = pred.size()[0]
        mean, log_var = pred.chunk(2, dim = 1)
        log_var = log_var.reshape(num_examples, -1)
        
        length = target.shape[1] # (B, L, D) <<-- take L
        max_logvar = self.max_logvar[:, :length]
        min_logvar = self.min_logvar[:, :length]
    
        log_var = max_logvar - F.softplus(max_logvar - log_var)
        log_var = min_logvar + F.softplus(log_var - min_logvar)

        log_var = log_var.reshape(*(target.shape))
        

        inv_var = torch.exp(-log_var)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.pow(mean - target, 2) * inv_var * (weight * loss_weight))
            var_loss = torch.mean(log_var * (weight * loss_weight))
            total_loss = mse_loss + var_loss
        else:
            mse_loss = torch.mean(torch.pow(mean - target, 2))
            total_loss = mse_loss
        if isinstance(loss_weight, int):
            total_loss += 0.01 * torch.mean(max_logvar) - 0.01 * torch.mean(min_logvar)
        else:
            total_loss += 0.01 / num_examples * torch.mean(max_logvar.reshape(1, *(mean.shape[1:])) * loss_weight) \
                        - 0.01 / num_examples * torch.mean(min_logvar.reshape(1, *(mean.shape[1:])) * loss_weight)

        return torch.mean(total_loss)