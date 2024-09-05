from timm.models.swin_transformer import SwinTransformer
from .backbone_config import SwinConfig
import torch.nn as nn
import argparse


from ..SignalModel import SignalBase
from einops import rearrange

class Swin2DSignalBase(SignalBase):

    
    @staticmethod
    def build_backbone_config(config:SwinConfig):

        config = argparse.Namespace(
            img_size   =config.img_size,      # [896,672]
            depths     =config.depths,        # [2,2,14,2]
            in_chans   =config.in_chans,
            window_size=config.window_size,   # input_size是window_size、patch_size和8(2**num_layer)的公倍数
            patch_size =config.patch_size,    # 4
            embed_dim  =config.embed_dim,     # 128
            num_heads  =config.num_heads,     # [4, 8, 16, 32]
            num_classes=0, ### <--return the feature
        )
        return config
    
    def build_backbone(self, config):
        config = self.build_backbone_config(config)
        self.backbone= SwinTransformer(**vars(config))
        return self.backbone
                            
    def get_key_token_in_parallel_mode(self, inputs_embeds):
        return inputs_embeds
    
    def get_kernel_output(self, x):
        """
        You should implment this for different architecture
        """

        x = self.backbone.forward_features(x)  # -> (B, hidden_size)
        x = x.unsqueeze(1) #(B,1, hidden_size)
        return x



class Swin2DSlidePred(Swin2DSignalBase):
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