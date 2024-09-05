

import torch
import torch.nn as nn
from .FlashAttn1D import SignalTransformer
from .backbone_config import ViTConfig
from einops import rearrange

import numpy as np  
import argparse
from trace_utils import print0


from ..SignalModel import SignalBase
from einops import rearrange
def get_absoluate_embedding(d_model, max_len=10000):
    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = (torch.arange(0, d_model, 2).float()* -(np.log(10000.0) / d_model)).exp()

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    return pe


class TransformerSignalBase(SignalBase):

    @staticmethod
    def build_backbone_config(args:ViTConfig):
        if args.signal_patch_size == 1:
            signal_length = args.sequence_length_in_backbone
            wave_channel  = args.hidden_size
        else:
            print0(f""" 
                   WARNING: you are try to use build-in patch of ViT, 
                   make sure your embedder output is correct.
                   And I will fix the wave channel you input to {args.hidden_size} (you need implement yourself for orgin input)
                   """)
            signal_length = args.sequence_length_in_backbone 
            wave_channel  = args.hidden_size
        
        config = argparse.Namespace(
            signal_length=signal_length, 
            patch_size=args.signal_patch_size, # <-- this mean we will not patch the sequence in transformer
            in_chans=wave_channel,
            embed_dim=args.hidden_size,
            depth=args.num_hidden_layers,
            num_heads=args.num_heads,

            use_flash_attn=args.use_flash_attn,
            rotary_emb_dim=args.rotary_emb_dim,
            disable_bias=args.disable_all_bias,
            num_classes=1, 
            qkv_bias=False,
            class_token=False,  # <---False
            norm_layer=None,
            act_layer=None,
            fused_bias_fc=False,
            fused_mlp=False,
        )
        return config
    
    def build_backbone(self, args):
        config  = self.build_backbone_config(args)
        self.backbone= SignalTransformer(**vars(config))
        self.backbone.head = nn.Identity()

        if args.rotary_emb_dim > 0:
            print0("use rotary embedding, disable global position embedding")
            self.backbone.pos_embed = 0
        else:
            self.backbone.pos_embed = torch.nn.Parameter(get_absoluate_embedding(args.hidden_size)[:, :self.backbone.embed_len],requires_grad=False)
        return self.backbone
                            
    def get_kernel_output(self, x):
        """
        You should implment this for different architecture
        """
        x = rearrange(x, 'B D L -> B L D')  # (B, 3, 6000) -> (B, 6000, 3)
        x = self.backbone.forward_features(x, all_tokens=True)  # -> (B, L, hidden_size)
        return x


class ViTSlidePred(TransformerSignalBase):
    @staticmethod
    def get_key_token_in_parallel_mode(inputs_embeds):
        return inputs_embeds
    
    def collect_kernel_output(self, outputs):
        hidden_states = outputs.last_hidden_state
        if self.config.Predictor.merge_token == 'average': fea = hidden_states.mean(1, keepdims=True)
        elif self.config.Predictor.merge_token == 'last' : fea = hidden_states[:, -1:, :]
        elif self.config.Predictor.merge_token == 'first': fea = hidden_states[:, 0:1, :]
        else:
            raise ValueError("merge_token only support average, last, first")
        
        downstream_feature = {}
        for key in self.predictor.keys():
            if key in ['findP', 'findS', 'findN']:
                downstream_feature[key] = (fea, hidden_states)
            else:
                downstream_feature[key] = fea
        return downstream_feature #past_key_values, fea, hidden_states