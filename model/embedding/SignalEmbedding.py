import numbers
import torch.nn.functional as F
from torch.nn.modules.conv import Tensor
import torch
import torch.nn as nn
from .layers.wavelets_pytorch.transform import WaveletTransInTorch
from .layers.wavelets_pytorch.wavelets import Morlet
from einops import rearrange
import math
try:
    from rotary_embedding_torch.rotary_embedding_torch import (
        RotaryEmbedding,
        rotate_half
    )
except:
    pass

class SymmetricConv1d(nn.Conv1d):
    def forward(self, input: Tensor) -> Tensor:
        weight = (self.weight + self.weight.flip(-1))/2
        return self._conv_forward(input, weight, self.bias)

class AntiSymmetricConv1d(nn.Conv1d):
    def forward(self, input: Tensor) -> Tensor:
        weight = (self.weight - self.weight.flip(-1))/2
        return self._conv_forward(input, weight, self.bias)

CNNModule={
    "symmetry": SymmetricConv1d,
    "antisymmetry": AntiSymmetricConv1d,
    "vallina": nn.Conv1d
}

class MultiScalerFeature(nn.Module):
    """
    the goal is use symmetric slide windows extract the signal features. 
    The symmetric actually implies smooth the high frequency signal. 
    """

    def __init__(self, in_feat, hidden_feat, out_feat, scalers=[3], abs_feature=True, stride = 1, cnn_type = 'symmetry'):
        super().__init__()
        for i in scalers:
            assert i % 2 == 1, f"the scalers must be odd, given {scalers}"
        self.abs_feature = abs_feature
        self.stride = stride
        self.embedders = nn.ModuleList()
        for scaler in scalers:
            padding = (scaler-stride)//2
            if (scaler-stride)%2 == 1:
                print(f"scaler - stride must mod 2, given scale={scaler} and stride={stride}, pass")
                continue
            if scaler < stride:
                print(f"scale must large then stride , given scale={scaler} and stride={stride}, pass")
                continue
            self.embedders.append(CNNModule[cnn_type](
                in_feat, hidden_feat, kernel_size=scaler, stride=stride, padding=padding, bias=False))
        assert len(self.embedders) > 0, "non feature level createded"

        self.feature_mixing = nn.Linear(int(self.abs_feature)*in_feat + hidden_feat*len(self.embedders), out_feat, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        def init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Linear)):
                torch.nn.init.constant_(m.weight, 1/m.weight.shape[1])
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        self.apply(init_weights)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # ( B, L, 3) -> (B, 3, L)
        abs_feature = [x.abs()[...,::self.stride]] if self.abs_feature else []
        features = torch.concatenate(abs_feature + [embedder(x) for embedder in self.embedders], 1)  # (B, 3, L) -> (B, s*N, L)

        features = features.permute(0, 2, 1)  # (B, N, L) -> (B, L, s*N)
        features = self.feature_mixing(features)  # (B, L, s*N) -> (B, L, N)
        return features


class UnitLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(
            normalized_shape)  # type: ignore[arg-type]
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(
            self.normalized_shape, **factory_kwargs))
        #self.register_buffer('weight', weight)

        bias = torch.zeros(self.normalized_shape, **factory_kwargs)
        self.register_buffer('bias', bias)

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps},'.format(**self.__dict__)

class First_Derivative_Layer(torch.nn.Module):
    '''
    The input dimension support 1d, 2d, 3d
    use high dimention conv is faster
        B=2
        P=4
        a=torch.randn(B,P,3,32,64).cuda()
        layer=  First_Derivative_Layer(dim=3).cuda()
        runtime_weight=layer.runtime_weight

        x = torch.conv3d(a.flatten(0,1).unsqueeze(1),runtime_weight).reshape(*a.shape[:-1],-1)
        --> 34.8 µs ± 130 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

        x2 = torch.conv1d(a.flatten(0,-2).unsqueeze(1),runtime_weight[0,0]).reshape(*a.shape[:-1],-1)
        --> 43.1 µs ± 1.63 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    '''
    def __init__(self,position=-1,dim=2,mode='five-point-stencil',pad_mode='circular',intervel=torch.Tensor([1])):
        super().__init__()
        self.postion=position
        self.dim    =dim
        self.mode   =mode
        self.pad_mode=pad_mode
        self.intervel= intervel
        self.conv_engine = [torch.conv1d, torch.conv2d, torch.conv3d][self.dim-1]
        if self.mode=='five-point-stencil':
             self.weight = torch.nn.Parameter(torch.Tensor([1/12,-8/12,0,8/12,-1/12]),requires_grad=False)
             self.pad_num = 2
        elif self.mode=='three-point-stencil':
             self.weight = torch.nn.Parameter(torch.Tensor([-1/2,0,1/2]),requires_grad=False)
             self.pad_num = 1
        elif isinstance(self.mode, int):
             self.weight = torch.nn.Parameter(torch.randn(self.mode)*0.01)
             self.pad_num     = self.mode
        else:
            raise NotImplementedError(f"the self.mode must be five-point-stencil or three-point-stencil or a int number to activate trainable derivate")
        padtuple = [0]*self.dim*2
        padtuple[(-self.postion-1)*2]   = self.pad_num
        padtuple[(-self.postion-1)*2+1] = self.pad_num
        self.padtuple = tuple(padtuple)

    @property
    def runtime_weight(self):
        if isinstance(self.mode, int):
            weight = torch.cat([-torch.flip(self.weight,(0,)),F.pad(self.weight,(1,0))])
        else:
            weight = self.weight
        return weight[(None,)*(self.dim + 1)].transpose(self.postion,-1)
    
    def forward(self, x):
        expandQ=False
        if len(x.shape) == self.dim + 1 and x.shape[1]!=1:
            x = x.unsqueeze(1)
            expandQ = True
        assert len(x.shape) == self.dim + 2
        # if only x dim, then the input should be (Batch, 1 , x)
        # if (x,y) pannel, then the input should be (Batch, 1 , x , y)
        # if (x,y,z) pannel, the the input should be (Batch, 1 , x , y , z)
        x = self.conv_engine(F.pad(x, self.padtuple, mode=self.pad_mode),self.runtime_weight)
        x = x/self.intervel.to(x.device)

        return x.squeeze(1) if expandQ else x

class Second_Derivative_Layer(torch.nn.Module):
    def __init__(self,position=(-2,-1),mode='nine-point-stencil',pad_mode='circular',intervel=torch.Tensor([1])):
        super().__init__()
        self.postion=position
        self.mode   =mode
        self.pad_mode=pad_mode
        self.intervel= intervel
        if self.mode=='nine-point-stencil':
            self.weight = torch.nn.Parameter(torch.Tensor([[1/3, 1/3,1/3],
                                                           [1/3,-8/3,1/3],
                                                           [1/3, 1/3,1/3]]),requires_grad=False)
            self.pad_num = 1
        else:
            raise NotImplementedError(f"the self.mode must be nine-point-stencil")

    @property
    def runtime_weight(self):
        return self.weight[None,None] #(1,1,3,3)
    def forward(self, x):
        x = x.transpose(-2,self.postion[0])
        x = x.transpose(-1,self.postion[1])
        
        oshape = x.shape
        x = x.flatten(0,-3).unsqueeze(1)
        x = F.pad(x, (self.pad_num,self.pad_num,self.pad_num,self.pad_num), mode=self.pad_mode)
        x = torch.conv2d(x,self.runtime_weight)
        x = x/self.intervel.to(x.device)
        x = x.reshape(*oshape)
        x = x.transpose(-1,self.postion[1])
        x = x.transpose(-2,self.postion[0])
        return x

class ContinusWaveletDecomposition(nn.Module):
    """
    the goal is use symmetric slide windows extract the signal features. 
    The symmetric actually implies smooth the high frequency signal. 
    """

    def __init__(self, length, dt, dj, total_scale_num=None):
        super().__init__()
        wavelet = Morlet(w0=6)
        self.wavelet = WaveletTransInTorch(length, dt, dj, wavelet, unbias=False, total_scale_num = total_scale_num,
                                           cuda=True #<---not any work
                                           )
        

    def forward(self, x):
        assert len(x.shape)==3 # (B, L, 3)
        B, L, D = x.shape
        with torch.no_grad():
            x = rearrange(x, 'B L D -> (B D) L')
            x = self.wavelet.power(x)
            x = rearrange(x, '(B D) N T L -> B (D T) N L', L=L, D=D)
        return x

class AddRotaryPosition(nn.Module):
    """
    USE sin/cos way embedding the position
    we assume the input is fixed length, thus we can use the fixed length position embedding, thus we can firstly generate the position embedding
    """
    def __init__(self, width, height, dim,  max_freq=256,):
        super().__init__()
        pos_emb = RotaryEmbedding(
            dim = dim,
            freqs_for = 'pixel',
            max_freq = max_freq
        )
        freqs = pos_emb.get_axial_freqs(width, height)
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)
        self.width = width 
        self.height = height
    def forward(self, x):
        assert x.shape[-2:] == (self.width, self.height)
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.apply_rotary_emb((self.freqs_cos, self.freqs_sin), x)
        x = rearrange(x, 'B H W C -> B C H W')
        return x
    
    @staticmethod
    def apply_rotary_emb(freqs, t, start_index = 0, scale = 1., seq_dim = -2):
        assert t.ndim == 4 # (B, H, W, C)
        freqs_cos, freqs_sin = freqs
        rot_dim   = freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[...,start_index:end_index], t[..., end_index:]
        t = (t * freqs_cos * scale) + (rotate_half(t) * freqs_sin * scale)
        return torch.cat((t_left, t, t_right), dim = -1)

    def __repr__(self):
        return super().__repr__()[:-2] + f"({self.width}, {self.height})"