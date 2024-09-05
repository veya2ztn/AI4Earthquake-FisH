from ..backbone_config import BackboneConfig
from dataclasses import dataclass
from simple_parsing import field
from config.utils import get_compare_namespace_trees, get_default_config
from typing import Optional

@dataclass
class ViTConfig(BackboneConfig):
    architecture: str = "ViT"
    signal_patch_size: int = field(default=1)
    use_flash_attn: bool = field(default=True)
    rotary_emb_dim: int  = field(default=64)
    preset_architecture = False #<---must be False, otherwise may goes into infinity loop when call type(self)()
    disable_all_bias: bool = field(default=False)
    sequence_length_in_backbone:Optional[int] = None
    def __post_init__(self):
        super().__post_init__()
        if not self.preset_architecture: return
        assert self.sequence_length_in_backbone is not None, "sequence_length_in_backbone must be assigned, when use ViT"
        # for key in ['use_lm_decay','disable_all_bias']:
        #     assert getattr(self, key) == getattr(type(self), key), f"""
        #     For fixed backbone config, you are now allowed custom the backbone config, below is the compare result:\n{get_compare_namespace_trees(self, type(self)())}
        #     """

@dataclass
class PresetViTConfig(ViTConfig):
    preset_architecture = True
    backbone_type = None
    
@dataclass
class BuildInPatch(PresetViTConfig):
    signal_patch_size: int = 3
    backbone_type = "InPatch"
    

@dataclass
class EmbedderPatch(PresetViTConfig):
    signal_patch_size: int = 1
    backbone_type = "OutPatch"

#########################################################################


@dataclass
class Model10M(BackboneConfig):
    hidden_size:       int = 448
    intermediate_size: int = 0 # <--- 0 means no need, useless 
    num_hidden_layers: int = 4
    attention_hidden_size: int = 0 # <--- 0 means no need, useless 
    num_heads:int =  16
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "10M"

@dataclass
class Model15M(BackboneConfig):
    hidden_size:       int = 512
    intermediate_size: int = 0 # <--- 0 means no need, useless 
    num_hidden_layers: int = 4
    attention_hidden_size: int = 0 # <--- 0 means no need, useless 
    num_heads:int =  16
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "15M"

@dataclass
class Model60M(BackboneConfig):
    hidden_size:       int = 1024
    intermediate_size: int = 0 # <--- 0 means no need, useless 
    num_hidden_layers: int = 4
    attention_hidden_size: int = 0 # <--- 0 means no need, useless 
    num_heads:int =  16
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "60M"

@dataclass
class Model120M(BackboneConfig):
    hidden_size:       int = 1024
    intermediate_size: int = 0 # <--- 0 means no need, useless 
    num_hidden_layers: int = 8
    attention_hidden_size: int = 0 # <--- 0 means no need, useless 
    num_heads:int =  16
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "120M"

@dataclass
class Model180M(BackboneConfig):
    hidden_size:       int = 1024
    intermediate_size: int = 0 # <--- 0 means no need, useless 
    num_hidden_layers: int = 12
    attention_hidden_size: int = 0 # <--- 0 means no need, useless 
    num_heads:int =  16
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "180M"

@dataclass
class Model240M(BackboneConfig):
    hidden_size:       int = 1024
    intermediate_size: int = 0 # <--- 0 means no need, useless 
    num_hidden_layers: int = 16
    attention_hidden_size: int = 0 # <--- 0 means no need, useless 
    num_heads:int =  16
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "240M"

@dataclass
class Model40M(BackboneConfig):
    hidden_size:       int = 768
    intermediate_size: int = 0 # <--- 0 means no need, useless 
    num_hidden_layers: int = 4
    attention_hidden_size: int = 0 # <--- 0 means no need, useless 
    num_heads:          int =  16
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "40M"

@dataclass
class EP10M(Model10M,EmbedderPatch): pass

@dataclass
class EP15M(Model15M,EmbedderPatch): pass

@dataclass
class EP40M(Model40M,EmbedderPatch): pass

@dataclass
class EP60M(Model60M,EmbedderPatch): pass

@dataclass
class EP120M(Model120M,EmbedderPatch): pass

@dataclass
class EP180M(Model180M,EmbedderPatch): pass
@dataclass
class EP240M(Model240M,EmbedderPatch): pass

@dataclass
class IP10M(Model10M,BuildInPatch): pass

@dataclass
class IP15M(Model15M,BuildInPatch): pass
