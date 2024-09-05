
from dataclasses import dataclass
from .backbone_config import *
from ...predictor.predictor_config import SlideWindowPredictorConfig
from ...embedding.embedding_config import WaveletTanhPhaseConfig
from ...model_arguements import SignalModelConfig


############################################
########### Slide Model ####################
############################################
@dataclass
class SlideConfig(SignalModelConfig):
    model_type: str = 'Swin2DSlidePred'

@dataclass
class WVLetSwin30M(SlideConfig):
    Embedding: WaveletTanhPhaseConfig
    Backbone:  SWIN30M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'WVLetSwinW30M'
    revise_backbone_name: str = 'WVLetSwin'

@dataclass
class WVLetSwin100M(SlideConfig):
    Embedding: WaveletTanhPhaseConfig
    Backbone:  SWIN100M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'WVLetSwinW100M'
    revise_backbone_name: str = 'WVLetSwin'