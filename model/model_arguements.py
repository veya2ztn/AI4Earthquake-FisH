from dataclasses import dataclass
from typing import Optional, Union, List,Tuple
from simple_parsing import ArgumentParser, subgroups, field
from simple_parsing.helpers.serialization import save
from transformers.configuration_utils import PretrainedConfig
from .embedding.embedding_config import EmbeddingConfig, WaveletEmbeddingConfig
from .signal_model.backbone_config import BackboneConfig
from .signal_model.SWIN.backbone_config import SwinConfig
from .predictor.predictor_config import PredictorConfig

import os
import inspect
import importlib
import dataclasses
from trace_utils import print0
def build_config_pool(module_path):
    module = importlib.import_module(module_path)
    config_pool = {}
    
    for name, obj in inspect.getmembers(module):
        if dataclasses.is_dataclass(obj) and obj.__module__ == module.__name__ and 'model_config_name' in obj.__annotations__:
            config_pool[obj.model_config_name] = obj
    return config_pool

nickname_map = {
        "RetNetDecay_SignalSeaL1.512.1024.256.16.4": 'Goldfish.10M.L1',
        "RetNetDecay_SignalSeaL1.512.1024.256.16.8": 'Goldfish.20M.L1',

        "RetNetDecay_SignalSeaL1.3.512.1024.256.8": 'Goldfish.20M.L1',
        'RetNetDecay_SignalSeaL1.3.512.1024.256.4': 'Goldfish.10M.L1',

        'RetNetDecay_SignalSea.3.512.1024.256.1': 'Goldfish.2M.Sea',
        'RetNetDecay_SignalSea.512.1024.256.16.1': 'Goldfish.2M.Sea',
        'RetNetDecay_SignalSea.3.512.1024.256.4': 'Goldfish.10M.Sea',
        'RetNetDecay_SignalSea.512.1024.256.16.4': 'Goldfish.10M.Sea',
            
        'RetNetDecay_SignalSea.3.768.512.256.1': 'Goldfish.2M_Z.Sea',
        'RetNetDecay_SignalSeaL1.3.768.512.256.4': 'Goldfish.10M_Z.L1',
        'RetNetDecay_SignalSea.3.768.512.256.4': 'Goldfish.10M_Z.Sea',
        "RetNetDecay_SignalSea.768.512.256.16.4":'Goldfish.10M_Z.Sea',
        "RetNetDecay_SignalSea.768.512.256.16.1":'Goldfish.2M_Z.Sea',
        "RetNetDecay_SignalSeaL1.768.512.256.16.4":'Goldfish.10M_Z.L1',
            
        "RetNetDecay_ReLU.3.512.1024.256.4":'Goldfish.ReLU.10M',
        "RetNetDecay_NoBias.3.512.1024.256.4":'Goldbird.10M',
}

@dataclass
class FreezeConfig:
    freeze_embedder: Optional[str] = None
    freeze_backbone: Optional[str] = None
    freeze_downstream: Optional[str] = None

@dataclass
class SignalModelConfig(PretrainedConfig):
    
    Embedding   : EmbeddingConfig
    Backbone    : BackboneConfig
    Predictor   : PredictorConfig
    

    model_type  : str 
    model_config_name : Optional[str]=None
    pruned_heads: bool = False ## <--- only for pass the class for transformers Pretrained model class
    is_composition = True
    nickname: Optional[str] = None
    trained_batch_size: Optional[int] = None
    train_on: Optional[str] = None
    revise_backbone_name: Optional[str] = None
    use_whole_layer_output: bool = False
    report_error: bool = True

    @property
    def name(self):
        if self.model_config_name is None:
            raise NotImplementedError
        return self.nickname

    @property
    def signal_property(self):
        return None

    def build_nick_name(self):
        
        nick_name = []
        if self.revise_backbone_name is not None and self.revise_backbone_name:
            nick_name.append(f"{self.revise_backbone_name}.{self.Backbone.size_type}")
        elif self.Backbone.backbone_nickname:
            nick_name.append(self.Backbone.backbone_nickname)
        else:
            raise NotImplementedError
        
        if self.signal_property:
            nick_name.append(self.signal_property)
        if self.Embedding.embedding_nickname:
            nick_name.append(self.Embedding.embedding_nickname)

        if len(nick_name)>=1:
            nick_name = ".".join(nick_name)
        else:
            nick_name = ""
        if self.nickname != nick_name:
            print0(f'WARNING: detect nickname conflict, change nickname to {nick_name} from {self.nickname}')
        self.nickname = nick_name

    def __post_init__(self):
        assert self.model_config_name is not None, "Since the model_config_name is None, it seem you are not assign any model by --model [model] "
        if isinstance(self.Embedding, WaveletEmbeddingConfig) and isinstance(self.Backbone, SwinConfig):
            self.Embedding.embedding_size = self.Backbone._in_chans = self.Backbone.embed_dim//2
            now_w = self.Embedding.wave_num_scales
            now_h = self.Embedding.wave_length
            for i,s in enumerate(self.Embedding.downsample_rate_list):
                s = int(s)
                k = 2*s - 1
                p = s - 1
                now_w = now_w // s
                now_h = now_h // s
            self.Backbone._img_size = (now_w, now_h)
            #print(self.Backbone._img_size)
        else:
            self.Embedding.disable_all_bias = self.Backbone.disable_all_bias
            self.Embedding.embedding_size   = self.Backbone.hidden_size
            self.Backbone.normalize_at_end  = not self.Predictor.normlize_at_downstream
        self.build_nick_name()
        
    def save_pretrained(self, save_directory):
        save(self,os.path.join(save_directory,"config.json"),indent=4)

