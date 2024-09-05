
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .TSLblock.layers.Embed import PositionalEmbedding, PatchEmbedding
from .SignalModel import SignalMagTimeDis
from .TSLblock.TimesNetBlock import TimesBlock

from .TSLblock.layers.Autoformer_EncDec import series_decomp
from typing import Dict, Optional
import argparse
import torch
import torch.nn as nn
class SignalSimpleEmbedding(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.projecter         = nn.Linear(in_chan, out_chan, bias=False)
        self.position_embedding = PositionalEmbedding(d_model=out_chan)
    def forward(self, x):
        x = self.projecter(x) + self.position_embedding(x)
        return x
import os
class TSLSignal(nn.Module, SignalMagTimeDis):

    def save_pretrained(self, path):
        if not os.path.exists(path):os.makedirs(path)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(path, 'checkpoints.bin'))

    def forward(
        self,
        status_seq: Optional[torch.LongTensor] = None,
        waveform_seq: Optional[torch.FloatTensor] = None,
        labels=None,
        get_prediction: Optional[bool] = None,
    ):
        status_seq    = self.deal_with_autoregress_sequence(status_seq)
        inputs_embeds = self.get_composed_input_embedding(status_seq, waveform_seq)
        _, hidden_states = self.kernel_forward(inputs_embeds)
        preded = self.downstream_prediction(hidden_states)
        
        target = {}
        
        if labels:
            for key, val in labels.items():
                if len(val.shape)>1 and self.min_used_length and val.shape[-1] > self.min_used_length:
                    target[key] = val[:,self.min_used_length:] # (B, L) ==> (B,100)
                elif len(val.shape) == 1:
                    target[key] = val.unsqueeze(1) # ==> (B,1)
                else:
                    target[key] = val
            loss, error_record, prediction = self.evaluate_error(target, preded,get_prediction=get_prediction)
        else:
            loss = error_record = prediction = None
        return {'loss':loss, 'error_record':error_record, 'prediction':prediction}


class TimesNetSignal(TSLSignal):
    def __init__(self, args, downstream_pool):
        super().__init__()
        self.args = args
        self.block_configs = argparse.Namespace(
            seq_len=args.max_length,
            label_len=0,
            pred_len=0,
            top_k=5,
            d_model=args.hidden_size,
            num_kernels=args.num_signal_kernel,
            d_ff=args.intermediate_size
        )

        self.sequence_embedding = SignalSimpleEmbedding(3, args.hidden_size)
        self.min_used_length = 100
        layers = []
        for _ in range(args.num_hidden_layers):
            layers.append(TimesBlock(self.block_configs))
            layers.append(nn.LayerNorm(args.hidden_size))

    
        layers.extend([
            Rearrange('B L D -> B D L'),
            nn.Linear(args.max_length,1,bias=False),
            Rearrange('B D L -> B L D'),
            
        ])

        self.backbone = nn.Sequential(*layers)
        self.build_downstring_task(args, downstream_pool)

    def get_composed_input_embedding(self, status_seq, waveform_seq):
        enc_out = self.sequence_embedding(waveform_seq)
        return enc_out

    def deal_with_autoregress_sequence(self,status_seq):
        return status_seq
    
    def kernel_forward(self, x):
        x = self.backbone(x) # (B, L, D)
        return None, x


class DLinearSignal(TSLSignal):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, args, downstream_pool, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super().__init__()
        self.seq_len      = args.max_length
        self.pred_len     = self.seq_len
        self.moving_avg   = 25
        self.decompsition = series_decomp(self.moving_avg)
        self.individual   = individual
        self.up_dim_chanel= nn.Linear(args.wave_channel,args.hidden_size)
        self.channels     = args.hidden_size
        if args.hidden_size > 128:
            print(f"use hidden_size={args.hidden_size} in DLinear, may create very large tensor")
        self.min_used_length = 100
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend    = nn.ModuleList()

            for _ in range(self.channels):
                for _list in [self.Linear_Seasonal, self.Linear_Trend]:
                    layer = nn.Linear(self.seq_len, self.pred_len)
                    layer.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                    _list.append(layer)
               
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        self.meger_along_time = nn.Sequential(
            Rearrange('B L D -> B D L'),
            nn.Linear(args.max_length,1,bias=False),
            Rearrange('B D L -> B L D'),
            
        )
        self.build_downstring_task(args, downstream_pool)


    def get_composed_input_embedding(self, status_seq, waveform_seq):
        
        return self.up_dim_chanel(waveform_seq)

    def deal_with_autoregress_sequence(self, status_seq):
        return status_seq

    def kernel_forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1) # (B, L, D)
        x = self.meger_along_time(x)
        return None,x
    
from .TSLblock.PatchTSTBlock import  PatchTST_Encoder
class PatchTSTSignal(TSLSignal):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, args, downstream_pool, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.min_used_length = 100
        padding = stride
        configs = argparse.Namespace(
            seq_len=args.max_length,
            label_len=0,
            pred_len=0,
            top_k=5,
            d_model=args.hidden_size,
            num_kernels=args.num_signal_kernel,
            d_ff=args.intermediate_size,
            dropout = 0,
            e_layers=args.num_hidden_layers,
            factor = 1,
            output_attention=False,
            n_heads= 8,
            activation='gelu'
        )
        # patching and embedding
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = PatchTST_Encoder(configs)

        # Prediction Head
        self.head_nf = int((configs.seq_len - patch_len) / stride + 2)


        self.meger_along_time = nn.Sequential(
            Rearrange('B L P D -> B D (L P)'),
            nn.Linear(self.head_nf*3, 1, bias=False),
            Rearrange('B D L -> B L D'),

        )
        self.build_downstring_task(args, downstream_pool)

    def get_composed_input_embedding(self, status_seq, waveform_seq):

        return waveform_seq

    def deal_with_autoregress_sequence(self, status_seq):
        return status_seq

    def kernel_forward(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc/stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        output = self.meger_along_time(enc_out)

        # Decoder
       
        return None,output
