from ..backbone_config import BackboneConfig
from dataclasses import dataclass
from simple_parsing import field
from config.utils import get_compare_namespace_trees, get_default_config
from typing import Optional, Union, List,Tuple

@dataclass
class SwinConfig(BackboneConfig):
    architecture= "Swin"
    backbone_type= 'swin'
    backbone_nickname= 'custom_swin'
    size_type='test'
    
    patch_size :int= 16 
    window_size:int= 8
    
    depths     : List[int]= field(default_factory=lambda: [2,4,4,2]) 
    num_heads  : List[int]= field(default_factory=lambda: [3, 6, 12 ,24])
    mlp_ratio  :float= 4. 
    qkv_bias   :bool= True
    drop_rate  :float= 0. 
    attn_drop_rate:float=0.
    drop_path_rate:float=0.1 
    patch_norm:bool=True

    _img_size  = None
    _in_chans  = None
    intermediate_size=None
    attention_hidden_size=None
    num_hidden_layers = None

    @property
    def in_chans(self):
        assert self._in_chans is not None, "please maunal set it in config streaming"
        return self._in_chans

    @property
    def img_size(self):
        assert self._img_size is not None, "please maunal set it in config streaming"
        return self._img_size
    
    @property
    def embed_dim(self):
        ## embed_dim * 2 ** i_layer == hidden_size
        dim_exploration = 2**(self.num_hidden_layers)
        assert self.hidden_size % dim_exploration==0
        return self.hidden_size//dim_exploration
    
    def __post_init__(self):

        
        self.num_hidden_layers = len(self.depths) - 1
        self.intermediate_size = None
        self.attention_hidden_size = None
        if not self.preset_model: 
            self.backbone_nickname = f"{self.backbone_type}.{self.hidden_size}.{self.intermediate_size}.{self.attention_hidden_size}.{self.num_heads}.{self.num_hidden_layers}"
            return
        self.backbone_nickname = f"{self.backbone_type}.{self.size_type}"
        for key in ['hidden_size']:
            assert getattr(self, key) == getattr(type(self), key), f"""
            For fixed backbone config, you are now allowed custom the backbone config, below is the compare result:\n{get_compare_namespace_trees(self, get_default_config(type(self)))}
            """
@dataclass
class PresetConfig(SwinConfig):
    preset_architecture = True
    backbone_type = None

@dataclass
class Model100M(BackboneConfig):
    hidden_size: int = 768*2
    depths     : List[int]= field(default_factory=lambda: [2,4,4,2]) 
    num_heads  : List[int]= field(default_factory=lambda: [3, 6, 12 ,24])
    mlp_ratio  :float= 4. 
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "100M"
    
@dataclass
class Model30M(BackboneConfig):
    hidden_size: int = 768
    depths     : List[int]= field(default_factory=lambda: [2,4,4,2]) 
    num_heads  : List[int]= field(default_factory=lambda: [3, 6, 12 ,24])
    mlp_ratio  :float= 4. 
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "30M"
@dataclass
class SWIN30M(Model30M,PresetConfig): pass
@dataclass
class SWIN100M(Model100M,PresetConfig): pass
