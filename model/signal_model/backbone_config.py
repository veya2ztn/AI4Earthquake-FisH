from dataclasses import dataclass,fields
from typing import Optional, Union, List, Tuple
from simple_parsing import field
from simple_parsing.helpers import Serializable
from config.utils import get_compare_namespace_trees, get_default_config




@dataclass
class BackboneConfig:
    hidden_size: int = field(default=256)
    intermediate_size: int = field(default=1024)
    num_hidden_layers: int = field(default=8)
    attention_hidden_size: int = field(default=256)
    num_heads: int = field(default=8)
    backbone_nickname: str = field(default=None)

    preset_model = False  #<--- if you dont want show it in finish config, just do it without :bool
    def __post_init__(self):
        
        if not self.preset_model: 
            self.backbone_nickname = f"{self.backbone_type}.{self.hidden_size}.{self.intermediate_size}.{self.attention_hidden_size}.{self.num_heads}.{self.num_hidden_layers}"
            return
        self.backbone_nickname = f"{self.backbone_type}.{self.size_type}"
        for key in ['hidden_size', 'intermediate_size','num_hidden_layers','attention_hidden_size','num_heads']:
            assert getattr(self, key) == getattr(type(self), key), f"""
            For fixed backbone config, you are now allowed custom the backbone config, below is the compare result:\n{get_compare_namespace_trees(self, get_default_config(type(self)))}
            """
        
        
    
     