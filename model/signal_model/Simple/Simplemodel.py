
from .SignalEmbedding import MultiScalerFeature
from .FlashAttn1D import SignalTransformer
import argparse
import torch
import torch.nn as nn
from typing import Dict, Optional

from .SignalModel import SignalMagTimeDis
import os
import torch
from einops import rearrange
import numpy as np

class SimpleSignalBase(nn.Module, SignalMagTimeDis):
    disable_all_bias = False
    

    def get_composed_input_embedding(self, status_seq, waveform_seq):
        enc_out = self.wave_embedding(waveform_seq)
        return enc_out

    def deal_with_autoregress_sequence(self, status_seq):
        return status_seq

    def kernel_forward(self, x):
        x = rearrange(x, 'B D L -> B L D')  # (B, 3, 6000) -> (B, 6000, 3)
        fea = self.backbone(x)  # -> (B, 1, hidden_size)
        return None, None, fea

    def forward(
        self,
        status_seq: Optional[torch.LongTensor] = None,
        waveform_seq: Optional[torch.FloatTensor] = None,
        labels=None,
        get_prediction: Optional[bool] = None,
    ):
        status_seq = self.deal_with_autoregress_sequence(status_seq)
        #print(waveform_seq.shape)
        inputs_embeds = self.get_composed_input_embedding(status_seq, waveform_seq)
        #print(inputs_embeds.shape)
        _, reduce_feature, feature = self.kernel_forward(inputs_embeds)
        preded = self.downstream_prediction(reduce_feature, feature)

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


class SimpleMSFASignal(SimpleSignalBase):
    '''
    If use this mode, the sequence lenght should be fixed since we use CNN. 
    The output for the hidden state should be (B, 1, D)
    '''

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
        
        self.backbone = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.Tanh(),
                                    nn.LayerNorm(args.hidden_size),
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.Tanh(),
                                    nn.LayerNorm(args.hidden_size))
        self.build_downstring_task(args, downstream_pool)
        
    def deal_with_autoregress_sequence(self,status_seq):
        return status_seq
    
    def get_composed_input_embedding(self, status_seq, waveform_seq):
        if len(status_seq.shape) == 2: status_seq = status_seq.unsqueeze(-1)
        _input = torch.cat([waveform_seq, status_seq], -1)
        enc_out = self.wave_embedding(_input)
        return enc_out

    def kernel_forward(self, x):
        fea = self.backbone(x)  # -> (B, L, hidden_size)
        x = fea.mean(1, keepdims=True)  # -> (B, 1, hidden_size)
        return None, x, fea
