
from dataclasses import dataclass
from .backbone_config import *
from ...predictor.predictor_config import RecurrentPredictorConfig, SlideWindowPredictorConfig
from ...embedding.embedding_config import (MSFReLUPhaseConfig, MSFTanhPhaseConfig, 
                                           MSFReLUDualConfig, MSFTanhDualConfig, 
                                           DFEConfig, MSFReLUPhaseConfig,DFETanhTrendConfig,
                                           MSFTanhTrendConfigV2,MSFTanhTrendConfig,WaveletTanhPhaseConfig)
from ...model_arguements import SignalModelConfig


############################################
########### Slide Model ####################
############################################
@dataclass
class SlideConfig(SignalModelConfig):
    model_type: str = 'ViTSlidePred'

@dataclass
class ViTMSFConfig(SlideConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  EmbedderPatch
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'ViTPredMSF'
    

@dataclass
class Victor10M(ViTMSFConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  EP10M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'Victor10M'
    revise_backbone_name: str = 'Victor'

@dataclass
class Victor15M(ViTMSFConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  EP15M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'Victor15M'
    revise_backbone_name: str = 'Victor'

@dataclass
class Hope60M(ViTMSFConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  EP60M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'Hope60M'
    revise_backbone_name: str = 'Hope'

@dataclass
class Hope15M(ViTMSFConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  EP15M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'Hope15M'
    revise_backbone_name: str = 'Hope'

@dataclass
class VSimple10M(ViTMSFConfig):
    Embedding: DFETanhTrendConfig
    Backbone:  IP10M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'VSimple10M'
    revise_backbone_name: str = 'VSimple'

@dataclass
class VSimple15M(ViTMSFConfig):
    Embedding: DFETanhTrendConfig
    Backbone:  IP15M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'VSimple15M'
    revise_backbone_name: str = 'VSimple'


@dataclass
class DreamW40M(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  EP40M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'DreamW40M'
    revise_backbone_name: str = 'Dream'

@dataclass
class DreamW60M(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  EP60M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'DreamW60M'
    revise_backbone_name: str = 'Dream'

@dataclass
class DreamW80M(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  EP60M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'DreamW60M'
    revise_backbone_name: str = 'Dream'

@dataclass
class DreamW120M(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  EP120M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'DreamW120M'
    revise_backbone_name: str = 'Dream'

@dataclass
class DreamW180M(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  EP180M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'DreamW180M'
    revise_backbone_name: str = 'Dream'

@dataclass
class DreamW240M(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  EP240M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'DreamW240M'
    revise_backbone_name: str = 'Dream'


@dataclass
class WVLetfishW40M(SlideConfig):
    Embedding: WaveletTanhPhaseConfig
    Backbone:  EP40M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'WVLetfishW40M'
    revise_backbone_name: str = 'WVLetfish'
