
import torch
import torch.nn as nn
import torch.nn.functional as F
from .SignalEmbedding import MultiScalerFeature, UnitLayerNorm, ContinusWaveletDecomposition, AddRotaryPosition
from .embedding_config import EmbeddingConfig, WaveletEmbeddingConfig
from trace_utils import print0
import numpy as np
from einops.layers.torch import Rearrange 
"""
Embed Seasonal(Centrilized)  with Symmetry CNN
Embed Trend(Non-Centrilized) with AntiSymmetry CNN
"""
def get_nonlinear(args):
    if args.embedding_nonlinear == 'tanh':
        nonlinear = nn.Tanh
    elif args.embedding_nonlinear == 'leaky_relu':
        nonlinear = nn.LeakyReLU  
    else:
        raise NotImplementedError(f"the nonlinear {args.embedding_nonlinear} is not implemented")
    return nonlinear
def generate_decreasing_powers_of_2(S, E, N):
    # Generate B numbers between 1 and A using linspace
    linear_space = np.linspace(S, E, N+1)
    
    # Round these numbers to the nearest power of 2
    powers_of_2_sequence = [2**np.floor(np.log2(x)).astype(int) for x in linear_space]
    powers_of_2_sequence[0] = S
    powers_of_2_sequence[-1]= E
    return powers_of_2_sequence

class AutoEmbedderBuilder:
    """
    You should use follow code to find the correct offset A.K.A the useful token for a sequence input
    --------------------
    from model.signal_model.Retnet.RetNet import RetNetRecurrent
    from model.model_arguements import SignalModelConfig
    from model.embedding.EmbedderBuilder import AutoEmbedderBuilder
    from model.embedding.embedding_config import EmbeddingConfig
    import torch.nn as nn
    class Embedder( AutoEmbedderBuilder,nn.Module):
        def __init__(self, args: EmbeddingConfig):
            super().__init__()
            self.build_signal_embedder(args)
        def build_signal_embedder(self, args):
            self.build_wave_embeder(args)


    from mltool.universal_model_util import init_weights

    from model.embedding.embedding_config import MSFTanhTrendConfigV2,MSFTanhTrendConfig

    import torch


    from dataclasses import dataclass
    from simple_parsing import field
    from config.utils import get_compare_namespace_trees, get_default_config


    @dataclass
    class MSFTanhTrendConfigV3(MSFTanhTrendConfig): 
        embedder: str = 'multifeature_cnn_V3'
        resolution: int = 1
        embedder_layers:int = 2

    args = MSFTanhTrendConfigV3(embedding_size=16)
    print(args)
    builder = Embedder(args)
    builder = builder.apply(init_weights)
    _=builder.eval()

    status_seq        = torch.randn(1,30,3)
    waveform_seq      = torch.randn(1,30,3)
    trend_seq         = torch.randn(1,30,3)
    aextra_waveform_seq= torch.randn(1,5,3)
    aextra_trend_seq   = torch.randn(1,5,3)

    embedding_stride       = 1
    useful_token_offset    = 6  + 3*(args.embedder_layers  - 2)
    cached_sequence_offset = 12 + 6*(args.embedder_layers  - 2)

    extra_waveform_seq = aextra_waveform_seq[:,:4]
    extra_trend_seq    = aextra_trend_seq   [:,:4]
    with torch.no_grad():

    #     inputs_embeds = builder.wave_embedding(waveform_seq)[0,:,0].detach().numpy().round(3)
    #     inputs_extra_embeds =  builder.wave_embedding(torch.cat([waveform_seq[:,:],extra_waveform_seq],1))[0,:,0].detach().numpy().round(3)
    #     extra_embeds =  builder.wave_embedding(torch.cat([waveform_seq[:,-12:],extra_waveform_seq],1))[0,:,0].detach().numpy().round(3)
        inputs_embeds      = builder.get_composed_input_embedding(status_seq, waveform_seq, trend_seq)[0,:,0].detach().numpy().round(3)
        inputs_extra_embeds= builder.get_composed_input_embedding(status_seq, torch.cat([waveform_seq[:,:],extra_waveform_seq],1), 
                                                                            torch.cat([trend_seq[:,:],extra_trend_seq],1))[0,:,0].detach().numpy().round(3)

    with torch.no_grad():
        
        
        whole_waveform_seq = torch.cat([waveform_seq[:,-cached_sequence_offset:],extra_waveform_seq],1) if cached_sequence_offset is not None else extra_waveform_seq
        whole_trend_seq    = torch.cat([   trend_seq[:,-cached_sequence_offset:],extra_trend_seq]   ,1) if cached_sequence_offset is not None else extra_trend_seq
        extra_embeds =  builder.get_composed_input_embedding(status_seq,whole_waveform_seq , whole_trend_seq)[0,:,0].detach().numpy().round(3)

    print(f"CHECK: When add {extra_waveform_seq.shape[1]} stamp, which token change")
    print(inputs_embeds)
    print(inputs_extra_embeds)
    print(extra_embeds)

    print("="*30)
    print("before token:",inputs_embeds[:-useful_token_offset])
    print("new token:"   ,inputs_embeds[-useful_token_offset:] )
    print("before token:",inputs_extra_embeds[:-useful_token_offset])
    print("new token:"   ,inputs_extra_embeds[-useful_token_offset:] )

    print("CHECK: how long history we need for get the upadted stamp")
    print(extra_embeds)

    print(f"the new token is:", extra_embeds[useful_token_offset:-useful_token_offset])

    assert len(inputs_embeds[:-useful_token_offset]) == len(inputs_extra_embeds[:-useful_token_offset]) - 1
    assert np.all(inputs_embeds[:-useful_token_offset] == inputs_extra_embeds[:-useful_token_offset-1])
    the_new_token = inputs_extra_embeds[-useful_token_offset-1]
    assert the_new_token == extra_embeds[useful_token_offset:-useful_token_offset]

    """
    @staticmethod
    def build_multifeature_cnn(args:EmbeddingConfig, add_abs_feature=True, cnn_type='symmetry'):
        layernorm = UnitLayerNorm if args.disable_all_bias else nn.LayerNorm
        nonlinear = get_nonlinear(args)
        layers = []
        layers.extend([MultiScalerFeature(args.wave_channel, args.msf_inner_dim,
                            args.msf_inner_dim, abs_feature=add_abs_feature, stride=3,  # 6000->2000
                            scalers=[3, 5, 7],  # args.msf_levels,
                            cnn_type=cnn_type),
                        nonlinear()]+ 
                        ([ nn.Dropout(args.embedding_dropout)] if args.embedding_dropout > 0  else []) + 
                        [layernorm(args.msf_inner_dim)])
        layers.extend([MultiScalerFeature(args.msf_inner_dim, args.msf_inner_dim,
                            args.embedding_size, abs_feature=False, stride=1,
                            scalers=[3, 5, 7],  # args.msf_levels,
                            cnn_type='vallina'),
                        nonlinear()]+ 
                        ([ nn.Dropout(args.embedding_dropout)] if args.embedding_dropout > 0  else []) + 
                        [layernorm(args.embedding_size),]
        )
        if args.embedder_layers > 2:
            for extra_layer_id in range(args.embedder_layers - 2):
                layers.extend([MultiScalerFeature(args.embedding_size, args.msf_inner_dim,
                                    args.embedding_size, abs_feature=False, stride=1,
                                    scalers=[3, 5, 7],  # args.msf_levels,
                                    cnn_type='vallina'),
                        nonlinear()]+ 
                        ([ nn.Dropout(args.embedding_dropout)] if args.embedding_dropout > 0  else []) + 
                        [layernorm(args.embedding_size)]
                )
        return nn.Sequential(*layers)

    @staticmethod
    def build_multifeature_cnn_V2(args:EmbeddingConfig, add_abs_feature=True, cnn_type='symmetry'):
        layernorm = UnitLayerNorm if args.disable_all_bias else nn.LayerNorm
        nonlinear = get_nonlinear(args)
        layers = []
        layers.extend([MultiScalerFeature(args.wave_channel, args.msf_inner_dim,
                            args.msf_inner_dim, abs_feature=add_abs_feature, stride=1,  # 6000->2000
                            scalers=[3, 5, 7],  # args.msf_levels,
                            cnn_type=cnn_type),
                        nonlinear()]+ 
                        ([ nn.Dropout(args.embedding_dropout)] if args.embedding_dropout > 0  else []) + 
                        [layernorm(args.msf_inner_dim)])
        layers.extend([MultiScalerFeature(args.msf_inner_dim, args.msf_inner_dim,
                            args.embedding_size, abs_feature=False, stride=3,
                            scalers=[3, 5, 7],  # args.msf_levels,
                            cnn_type='vallina'),
                        nonlinear()]+ 
                        ([nn.Dropout(args.embedding_dropout)] if args.embedding_dropout > 0  else []) + 
                        [layernorm(args.embedding_size)]
        )
        if args.embedder_layers > 2:
            for extra_layer_id in range(args.embedder_layers - 2):
                layers.extend([MultiScalerFeature(args.embedding_size, args.msf_inner_dim,
                                    args.embedding_size, abs_feature=False, stride=1,
                                    scalers=[3, 5, 7],  # args.msf_levels,
                                    cnn_type='vallina'),
                        nonlinear()]+ 
                        ([ nn.Dropout(args.embedding_dropout)] if args.embedding_dropout > 0  else []) + 
                        [layernorm(args.embedding_size)]
                )
        return nn.Sequential(*layers)

    @staticmethod
    def build_multifeature_cnn_V3(args:EmbeddingConfig, add_abs_feature=True, cnn_type='symmetry'):
        layernorm = UnitLayerNorm if args.disable_all_bias else nn.LayerNorm
        nonlinear = get_nonlinear(args)
        layers = []
        layers.extend([MultiScalerFeature(args.wave_channel, args.msf_inner_dim,
                            args.msf_inner_dim, abs_feature=add_abs_feature, stride=1,  # 6000->2000
                            scalers=[3, 5, 7],  # args.msf_levels,
                            cnn_type=cnn_type),
                        nonlinear()]+ 
                        ([ nn.Dropout(args.embedding_dropout)] if args.embedding_dropout > 0  else []) + 
                        [layernorm(args.msf_inner_dim)])
        layers.extend([MultiScalerFeature(args.msf_inner_dim, args.msf_inner_dim,
                            args.embedding_size, abs_feature=False, stride=1,
                            scalers=[3, 5, 7],  # args.msf_levels,
                            cnn_type='vallina'),
                        nonlinear()]+ 
                        ([nn.Dropout(args.embedding_dropout)] if args.embedding_dropout > 0  else []) + 
                        [layernorm(args.embedding_size)]
        )
        if args.embedder_layers > 2:
            for extra_layer_id in range(args.embedder_layers - 2):
                layers.extend([MultiScalerFeature(args.embedding_size, args.msf_inner_dim,
                                    args.embedding_size, abs_feature=False, stride=1,
                                    scalers=[3, 5, 7],  # args.msf_levels,
                                    cnn_type='vallina'),
                        nonlinear()]+ 
                        ([ nn.Dropout(args.embedding_dropout)] if args.embedding_dropout > 0  else []) + 
                        [layernorm(args.embedding_size)]
                )
        return nn.Sequential(*layers)

    @staticmethod
    def build_continues_wavelet_decompostion(args:WaveletEmbeddingConfig):
        layernorm = UnitLayerNorm if args.disable_all_bias else nn.LayerNorm
        nonlinear = get_nonlinear(args)
        layernorm = nn.BatchNorm2d
        layers = []
        layers.extend([ContinusWaveletDecomposition(args.wave_length,dt=args.wavelet_dt, dj=args.wavelet_dj, total_scale_num=args.wave_num_scales),
                    nn.LayerNorm((args.wave_num_scales, args.wave_length )),
                    AddRotaryPosition(args.wave_num_scales, args.wave_length,args.wave_channel,max_freq=16),])

        
        downsample_rate_list = args.downsample_rate_list
        embedding_dim_list = generate_decreasing_powers_of_2(args.wave_channel*2, args.embedding_size, len(downsample_rate_list))
        ### usually, this give a tensor (B, L, D, 2) and we
        now_w = args.wave_num_scales
        now_h = args.wave_length
        for i,s in enumerate(downsample_rate_list):
            s = int(s)
            k = 2*s - 1
            p = s - 1
            now_w = now_w // s
            now_h = now_h // s
            layers.extend( [nn.Conv2d(embedding_dim_list[i], embedding_dim_list[i+1], kernel_size=k, stride=s, padding=p, bias=not args.disable_all_bias),
                            nonlinear()]+ 
                            ([nn.Dropout(args.embedding_dropout)] if args.embedding_dropout > 0  else []) + 
                            [layernorm(embedding_dim_list[i+1]),
                            AddRotaryPosition(now_w, now_h,embedding_dim_list[i+1]//2)]
            )
        if not args.use2Dfeature:
            layers.append(Rearrange('B C W H -> B (W H) C'))
        return nn.Sequential(*layers)

    @staticmethod
    def simple_naive_multifeature(args:EmbeddingConfig):
        layernorm = UnitLayerNorm if args.disable_all_bias else nn.LayerNorm
        nonlinear = get_nonlinear(args)
        layers = []
        layers.extend([nn.Linear(args.wave_channel, args.embedding_size),nonlinear()]+ 
                      ([nn.Dropout(args.embedding_dropout)] if args.embedding_dropout > 0  else []) + 
                      [layernorm(args.embedding_size) if args.add_layer_norm else nn.Identity()],
                      [nn.Linear(args.embedding_size, args.embedding_size), nonlinear()]+ 
                      ([ nn.Dropout(args.embedding_dropout)] if args.embedding_dropout > 0  else []) + 
                      [layernorm(args.embedding_size) if args.add_layer_norm else nn.Identity()]
                )
        return nn.Sequential(*layers)
    
    def build_wave_embeder(self, args:EmbeddingConfig):
        self.embedder_config = args
        disable_all_bias = args.disable_all_bias
        layernorm = UnitLayerNorm if args.disable_all_bias else nn.LayerNorm
        if args.embedding_nonlinear == 'tanh':
            nonlinear = nn.Tanh
        elif args.embedding_nonlinear == 'leaky_relu':
            nonlinear = nn.LeakyReLU  
        else:
            raise NotImplementedError(f"the nonlinear {args.embedding_nonlinear} is not implemented")
        
        if self.embedder_config.embedder == 'multifeature_cnn':
            self.wave_embedding = self.build_multifeature_cnn(args, add_abs_feature=args.add_abs_feature, cnn_type=args.cnn_type)
            if self.embedder_config.used_information == 'waveform+trend':
                self.trend_embedding = self.build_multifeature_cnn(args, add_abs_feature=False, cnn_type=args.trend_cnn_type)
            assert self.embedder_config.resolution == 3, "the resolution must be 3"
            self.embedding_stride       = 3 
            self.useful_token_offset    = 4  + 3*(self.embedder_config.embedder_layers  - 2)
            self.cached_sequence_offset = 24 + 18*(self.embedder_config.embedder_layers  - 2)
            ### should mind that this rule only works for your multi feature is [3,5,7]
        elif self.embedder_config.embedder == 'multifeature_cnn_V2':
            self.wave_embedding = self.build_multifeature_cnn_V2(args, add_abs_feature=args.add_abs_feature, cnn_type=args.cnn_type)
            if self.embedder_config.used_information == 'waveform+trend':
                self.trend_embedding = self.build_multifeature_cnn_V2(args, add_abs_feature=False, cnn_type=args.trend_cnn_type)
            assert self.embedder_config.resolution == 3, "the resolution must be 3"
            self.embedding_stride       = 3
            self.useful_token_offset    = 2  + 3*(self.embedder_config.embedder_layers  - 2)
            self.cached_sequence_offset = 12 + 18*(self.embedder_config.embedder_layers  - 2)
            ### should mind that this rule only works for your multi feature is [3,5,7]
        elif self.embedder_config.embedder == 'multifeature_cnn_V3':
            self.wave_embedding = self.build_multifeature_cnn_V3(args, add_abs_feature=args.add_abs_feature, cnn_type=args.cnn_type)
            if self.embedder_config.used_information == 'waveform+trend':
                self.trend_embedding = self.build_multifeature_cnn_V3(args, add_abs_feature=False, cnn_type=args.trend_cnn_type)
            assert self.embedder_config.resolution == 1, "the resolution must be 1"
            self.embedding_stride       = 1
            self.useful_token_offset    = 6  + 3*(self.embedder_config.embedder_layers  - 2)
            self.cached_sequence_offset = 12 + 6*(self.embedder_config.embedder_layers  - 2)
            ### should mind that this rule only works for your multi feature is [3,5,7]
        elif self.embedder_config.embedder == 'directly_embedding':
            self.wave_embedding = self.simple_naive_multifeature(args)
            if self.embedder_config.used_information == 'waveform+trend':
                self.trend_embedding = self.simple_naive_multifeature(args)
            assert self.embedder_config.resolution == 1, "the resolution must be 1"
            self.embedding_stride       = 1
            self.useful_token_offset    = None ## [ whole token is valid]
            self.cached_sequence_offset = None
        elif self.embedder_config.embedder == 'wavelet_embedding':
            self.wave_embedding = self.build_continues_wavelet_decompostion(args)
            assert 'trend' not in self.embedder_config.used_information
            assert self.embedder_config.resolution == 1, "the resolution must be 1"
            self.embedding_stride       = 1
            self.useful_token_offset    = None ## [ whole token is valid]
            self.cached_sequence_offset = None
        else:
            raise NotImplementedError(f"the embedder {self.embedder_config.embedder} is not implemented")
        
        if self.embedder_config.used_information == 'waveform+trend':
            self.mixing = nn.Linear(args.embedding_size*2, args.embedding_size, bias=not disable_all_bias)



    def deal_with_autoregress_sequence(self, status_seq):
        raise NotImplementedError('you must deal_with_autoregress_sequence')
    
    def get_composed_input_embedding(self, status_seq, waveform_seq,trend_seq=None):
        if self.embedder_config.used_information == 'waveform':
            #assert status_seq is None, 'In the latest version, since we only use waveform signal, status_seq must be None' ## dont do this ~~~!!!
            enc_out = self.wave_embedding(waveform_seq)
            return enc_out
        elif self.embedder_config.used_information == 'waveform+status_seq':
            assert status_seq is not None, 'In the latest version, status_seq must be used'
            if len(status_seq.shape) == 2:status_seq = status_seq.unsqueeze(-1)
            _input = torch.cat([waveform_seq, status_seq], -1)
            enc_out = self.wave_embedding(_input)
            return enc_out
        elif self.embedder_config.used_information == 'waveform+trend':
            assert trend_seq is not None, 'trend_seq must be used'
            

            enc_out = self.wave_embedding(waveform_seq)
            trend_v = self.trend_embedding(trend_seq)
            
            enc_out = self.mixing(torch.cat([enc_out, trend_v], -1))
            return enc_out
        else:
            raise NotImplementedError('Only support only_waveform_seq or only_status_seq')
        
    def get_signal_embeddings(self, status_seq, waveform_seq):
        status_seq                 = self.deal_with_autoregress_sequence(status_seq)
        inputs_embeds              = self.get_composed_input_embedding(status_seq, waveform_seq)
        return inputs_embeds
    
   
    def get_cached_sequence(self, sequence):
        if sequence is None:return None
        if self.cached_sequence_offset is None: return None
        return sequence[:,-self.cached_sequence_offset:] ##
    

    def get_key_token_in_recurrent_mode(self, inputs_embeds):
        #assert inputs_embeds.shape[1] == 9, "the input_embeds should be (B,9,D), you given {}".format(inputs_embeds.shape)
        if self.useful_token_offset is None:return inputs_embeds
        return inputs_embeds[:,self.useful_token_offset:-self.useful_token_offset] ### every time we add 3 more stamp data and combine with last 24 and obtain the token list, due to the padding rules, only the 5th token is the new token    


    def get_key_token_in_parallel_mode(self, inputs_embeds):
        if self.useful_token_offset is None:return inputs_embeds
        return inputs_embeds[:,:-self.useful_token_offset]
    
    def pick_up_right_token_for_stride_mode(self, status_seq):
        return self.get_key_token_in_parallel_mode(status_seq[:,::self.embedding_stride])

    def freeze_embedder(self, freeze_mode, mode='train'):
        if freeze_mode is None:return
        if not freeze_mode: return 
        if freeze_mode == 'only_embedding':
            for param in self.parameters():
                param.requires_grad = False
            for module in self.modules():
                module.eval()  # Set the module to evaluation mode
            free_layers = [self.wave_embedding]
            if self.embedder_config.used_information == 'waveform+trend':
                free_layers += [self.mixing, self.trend_embedding]
            if mode == 'train':
                for free_layer in free_layers:
                    for param in free_layer.parameters():
                        param.requires_grad = True
                    free_layer.train()  # Set the module to evaluation mode
        elif freeze_mode == 'freeze_embedding':
            freeze_layers = [self.wave_embedding]
            if self.embedder_config.used_information == 'waveform+trend':
                freeze_layers += [self.mixing, self.trend_embedding]
            for freeze_layer in freeze_layers:
                for param in freeze_layer.parameters():
                    param.requires_grad = False
                freeze_layer.eval()  # Set the module to evaluation mode
        else:
            raise NotImplementedError(f"freeze mode {freeze_mode} not support")