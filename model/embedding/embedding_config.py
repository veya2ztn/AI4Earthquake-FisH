
from dataclasses import dataclass
from simple_parsing import field
from config.utils import get_compare_namespace_trees, get_default_config
from typing import List,Optional
import numpy as np
from trace_utils import print0
#######################################
######### Embedding Config ############
#######################################


@dataclass
class EmbeddingConfig:
    #vocab_size: int = 3,###  we are not a language model, so we don't need this
    embedder: str = None
    wave_channel:     int = None
    used_information: str = None
    embedding_size:   int = None ## <-- lets assign it from backbone config
    disable_all_bias: bool = False
    embedding_nickname: str = None
    preset_embedder = False #<---must be False, otherwise may goes into infinity loop when call type(self)()
    embedder_layers: int = 2
    embedding_dropout: float = 0.0
    resolution: int = 1
    
    def __post_init__(self):
        assert self.embedder is not None, "embedder must be assigned"
        assert self.wave_channel is not None, "wave_channel must be assigned"
        #assert self.embedding_size is not None, "embedding_size must be assigned"
        if self.used_information is None:
            if self.wave_channel == 3: self.used_information = 'waveform'
            elif self.wave_channel == 4: self.used_information = 'waveform+status_seq'
            else:
                raise NotImplementedError(f"wave_channel={self.wave_channel} for auto mode is not implemented yet")

def is_power_of_two(num):
    if num <= 0:
        return False
    return (num & (num - 1)) == 0

@dataclass
class WaveletEmbeddingConfig(EmbeddingConfig):
    embedder: str = 'wavelet_embedding'
    wave_channel: int = 3
    used_information: str = 'waveform'
    embedding_nonlinear: str = 'tanh'
    add_layer_norm : bool = False
    wavelet_dj: float = 0.125 ## (10/0.0125 = 800 scales)
    wavelet_dt: float = 1/50 ## 50 Hz
    wave_length: int = 3000
    embedder_layers: None # <-- not use anymore
    wave_num_scales : Optional[int] = None
    downsample_rate_list: str = "4,2,2"
    use2Dfeature: bool = False
    def __post_init__(self):
        super().__post_init__()
        assert self.wave_num_scales is not None, "wave_num_scales must be assigned"
        assert is_power_of_two(self.wave_length), "wave_length must be power of 2"
        assert is_power_of_two(self.wave_num_scales), "wave_num_scales must be power of 2"
        for t in self.downsample_rate_list.split(','):
            assert int(t)%2==0, "downsample_rate_list must be even"
        if isinstance(self.downsample_rate_list, str):
            self.downsample_rate_list = [int(t) for t in self.downsample_rate_list.split(',')]
        print0(f"please make sure the frequence for wavelet and your datasource is match, given wavelet_dt={self.wavelet_dt}")
        print0(f"please make sure the wave_length for wavelet and your datasource is match, given wavelet_dt={self.wave_length}")
        
@dataclass
class DFEConfig(EmbeddingConfig):        
    embedder: str = 'directly_embedding'
    embedding_nonlinear: str = field(default='leaky_relu')
    wave_channel: int = field(default=3)
    used_information: str = field(default='waveform')
    add_layer_norm : bool = field(default=False)
    
@dataclass
class DFETanhTrendConfig(EmbeddingConfig):        
    embedder: str = 'directly_embedding'
    embedding_nonlinear: str = field(default='tanh')
    wave_channel: int = field(default=3)
    used_information: str = field(default='waveform+trend')
    add_layer_norm : bool = field(default=True)


@dataclass
class MSFConfig(EmbeddingConfig):
    embedder: str = 'multifeature_cnn'
    num_signal_kernel: int = field(default=2)
    msf_inner_dim: int = field(default=64)
    embedding_nonlinear: str = 'tanh'
    resolution: int = 3
    preset_embedder = False
    add_abs_feature: bool = True
    cnn_type: str = field(default='symmetry', choices=['symmetry', 'vallina'])
    def __post_init__(self):
        super().__post_init__()
        if not self.preset_embedder: return
        for key in ['embedding_nonlinear','disable_all_bias']:
            assert getattr(self, key) == getattr(type(self), key), f"""
            For fixed backbone config, you are now allowed custom the backbone config, below is the compare result:\n{get_compare_namespace_trees(self, type(self)())}
            """

@dataclass
class MSFTanh(MSFConfig):
    embedding_nonlinear: str = field(default='tanh')
    preset_embedder = True 


@dataclass
class MSFTanhPhaseConfig(MSFTanh):
    used_information: str = field(default='waveform')
    wave_channel: int = field(default=3)

@dataclass
class WaveletTanhPhaseConfig(WaveletEmbeddingConfig):
    embedding_nonlinear: str = field(default='tanh')
    preset_embedder = True 
    used_information: str = field(default='waveform')
    wave_channel: int = field(default=3)
    

@dataclass
class MSFTanhDualConfig(MSFTanh): 
    used_information: str = field(default='waveform+status_seq')
    wave_channel: int = field(default=4)

@dataclass
class MSFTanhTrendConfig(MSFTanh): 
    used_information: str = field(default='waveform+trend')
    wave_channel:     int = field(default=3)
    trend_cnn_type:   str = field(default='antisymmetry', choices=['antisymmetry', 'vallina'])

@dataclass
class MSFTanhTrendConfigV2(MSFTanhTrendConfig): 
    embedder: str = 'multifeature_cnn_V2'

@dataclass
class MSFTanhTrendConfigV3(MSFTanhTrendConfig): 
    embedder: str = 'multifeature_cnn_V3'
    resolution: int = 1


@dataclass
class MSFReLU(MSFConfig):
    embedding_nonlinear: str = field(default='leaky_relu')
    preset_embedder = True 
    embedding_nickname = 'ReLU'


@dataclass
class MSFReLUPhaseConfig(MSFReLU):
    used_information: str = field(default='waveform')
    wave_channel: int = field(default=3)

@dataclass
class MSFReLUDualConfig(MSFReLU):
    used_information: str = field(default='waveform+status_seq')
    wave_channel: int = field(default=4)

@dataclass
class MSFReLUTrendConfig(MSFReLU): 
    used_information: str = field(default='waveform+trend')
    wave_channel:     int = field(default=3)
    trend_cnn_type:   str = field(default='antisymmetry', choices=['antisymmetry', 'vallina'])


@dataclass
class MSFReLUTrendConfigV3(MSFReLUTrendConfig): 
    embedder: str = 'multifeature_cnn_V3'
    resolution: int = 1

