from mltool.ModelArchi.FNONET.fnonet import AFNONet
from mltool.ModelArchi.FNONET.blocks import Block
from ..SignalModel import SignalMagTimeDis
from einops import rearrange
import torch.nn as nn
import os
from typing import Dict, Optional
import torch 
from functools import partial

class AFNONETSignalBase(nn.Module, SignalMagTimeDis):

    def get_composed_input_embedding(self, status_seq, waveform_seq):
        return waveform_seq

    def deal_with_autoregress_sequence(self, status_seq):
        return status_seq

    def kernel_forward(self, x):
        x = rearrange(x, 'B L D -> B D L')  # (B, 6000, 3) -> (B, 3, 6000)
        x = self.backbone.forward_features(
            x, all_tokens=False)  # -> (B, 1, hidden_size)
        return None, x

    def forward(
        self,
        status_seq: Optional[torch.LongTensor] = None,
        waveform_seq: Optional[torch.FloatTensor] = None,
        labels=None,
        get_prediction: Optional[bool] = None,
    ):
        status_seq = self.deal_with_autoregress_sequence(status_seq)
        inputs_embeds = self.get_composed_input_embedding(
            status_seq, waveform_seq)
        _, hidden_states = self.kernel_forward(inputs_embeds)
        preded = self.downstream_prediction(hidden_states)

        target = {}

        if labels:
            for key, val in labels.items():
                if len(val.shape) > 1 and self.min_used_length and val.shape[-1] > self.min_used_length:
                    # (B, L) ==> (B,100)
                    target[key] = val[:, self.min_used_length:]
                elif len(val.shape) == 1:
                    target[key] = val.unsqueeze(1)  # ==> (B,1)
                else:
                    target[key] = val
            loss, error_record, prediction = self.evaluate_error(
                target, preded, get_prediction=get_prediction)
        else:
            loss = error_record = prediction = None
        return {'loss': loss, 'error_record': error_record, 'prediction': prediction}

from .SignalEmbedding import MultiScalerFeature
class AFNONetSignalMSFAll(AFNONETSignalBase):
    def __init__(self, args, downstream_pool):
        super().__init__()
        self.min_used_length = 0
        self.wave_embedding = nn.Sequential(
            MultiScalerFeature(args.wave_channel, args.msf_inner_dim,
                               args.msf_inner_dim, abs_feature=True, stride=3,  # 6000->2000
                               scalers=[3, 5, 7],  # args.msf_levels,
                               cnn_type='symmetry'),
            nn.Tanh(),
            nn.LayerNorm(args.msf_inner_dim),
            MultiScalerFeature(args.msf_inner_dim, args.msf_inner_dim,
                               args.hidden_size, abs_feature=False, stride=1,
                               scalers=[3, 5, 7],  # args.msf_levels,
                               cnn_type='vallina'),
            nn.Tanh(),
            nn.LayerNorm(args.hidden_size),
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.backbone = nn.Sequential(*[Block(dim=args.hidden_size, mlp_ratio=4, drop=0,
                    drop_path=0,
                    norm_layer=norm_layer,
                    region_shape=(args.max_length//3,),
                    double_skip=False,
                    fno_blocks=args.num_heads,
                    fno_bias=False,
                    fno_softshrink=False) for i in range(args.num_hidden_layers)])
        self.build_downstring_task(args, downstream_pool)

    def kernel_forward(self, x):
        #x = torch.nn.functional.pad(x,(0,0,0,48))
        x = self.backbone(x)  # (B, L//P, hidden_size) -> (B, L//P, hidden_size)
        x = x.mean(1, keepdims=True)  # -> (B, 1, hidden_size)
        return None, x
    
    def get_composed_input_embedding(self, status_seq, waveform_seq):
        if len(status_seq.shape) == 2:
            status_seq = status_seq.unsqueeze(-1)
        _input = torch.cat([waveform_seq, status_seq], -1)
        enc_out = self.wave_embedding(_input)
        return enc_out

import argparse
class AFNONetSignalAll(AFNONETSignalBase):
    def __init__(self, args, downstream_pool):
        super().__init__()
        self.min_used_length = 0
        config = argparse.Namespace(
            img_size=(args.max_length,), patch_size=(args.signal_patch_size,), 
            in_chans=args.wave_channel, out_chans=args.wave_channel, 
            embed_dim=args.hidden_size, depth=args.num_hidden_layers, 
            fno_blocks=args.num_heads,build_head=False,
        )
        self.backbone = AFNONet(**vars(config))
        self.build_downstring_task(args, downstream_pool)

    def kernel_forward(self, x):
        x = rearrange(x, 'B L D -> B D L')  # (B, 6000, 3) -> (B, 3, 6000)
        x = self.backbone.forward_features(x)  # -> (B, L//P, hidden_size)
        for blk in self.backbone.blocks:x = blk(x)  # -> (B, L//P, hidden_size)
        x = x.mean(1, keepdims=True)  # -> (B, 1, hidden_size)
        return None, x
    
    def get_composed_input_embedding(self, status_seq, waveform_seq):
        if len(status_seq.shape)==2:status_seq=status_seq.unsqueeze(-1)
        return torch.cat([waveform_seq,status_seq],-1)