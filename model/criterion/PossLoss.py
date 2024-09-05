import torch
import torch.nn as nn
import torch.nn.functional as F
from .criterion_arguements import PosslossConfig

class Possloss(nn.Module):
    
    def __init__(self, config:PosslossConfig):
        self.config     = config
        self.max_logvar = torch.nn.Parameter((torch.ones( (1, config.final_sequence_length)).float() / 2) , requires_grad=True).to(self.device)
        self.min_logvar = torch.nn.Parameter((-torch.ones((1, config.final_sequence_length)).float() * 10), requires_grad=True).to(self.device)
        self.weight     = self.weight_stretagy('unified')

    def weight_stretagy(self, stretagy_name):
        assert stretagy_name == 'unified'
        weight  = 1
        return weight

    def forward(self, pred:torch.Tensor, target:torch.Tensor):
        # print(pred.shape, target.shape, self.max_logvar.shape, self.min_logvar.shape)
        inc_var_loss = self.config.inc_var_loss # kwargs.get("inc_var_loss", True)
        loss_weight  = self.config.loss_weight  # kwargs.get("weight", 1)
        weight  = self.weight
        
        if loss_weight is None:
            loss_weight = 1        
        num_examples  = pred.size()[0]
        mean, log_var = pred.chunk(2, dim = 1)
        log_var = log_var.reshape(num_examples, -1)

        max_logvar = self.max_logvar
        min_logvar = self.min_logvar
    
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