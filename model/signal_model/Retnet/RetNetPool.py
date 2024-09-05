
from dataclasses import dataclass
from .backbone_config import *
from ...predictor.predictor_config import RecurrentPredictorConfig, SlideWindowPredictorConfig
from ...embedding.embedding_config import (MSFReLUPhaseConfig, MSFTanhPhaseConfig, 
                                           MSFReLUDualConfig, MSFTanhDualConfig, 
                                           DFEConfig, MSFReLUPhaseConfig,
                                           MSFTanhTrendConfigV2,MSFTanhTrendConfigV3, MSFReLUTrendConfigV3, MSFTanhTrendConfig,
                                           WaveletTanhPhaseConfig)
from ...model_arguements import SignalModelConfig


############################################
########### Slide Model ####################
############################################
@dataclass
class SlideConfig(SignalModelConfig):
    model_type: str = 'RetNetSlidePred'
    Predictor: SlideWindowPredictorConfig
    @property
    def signal_property(self):
        return 'Slide'
    
@dataclass
class RetNetSlideMSFConfig(SlideConfig):
    Embedding: MSFTanhDualConfig
    Backbone:  Normal
    model_config_name:str = 'RetNetPred_MSF'
    
@dataclass
class RetNetSlidePhaseConfig(SlideConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Normal
    model_config_name:str = 'RetNetPred_MSF_Phase'


@dataclass
class PearlfishW10M(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k10M
    model_config_name:str = 'PearlfishW10M'
    revise_backbone_name: str = 'Pearlfish'

@dataclass
class PearlfishW20MA(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k20MA
    model_config_name:str = 'PearlfishW20M'
    revise_backbone_name: str = 'Pearlfish'

@dataclass
class PearlfishW40MA(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k40MA
    model_config_name:str = 'PearlfishW40M'
    revise_backbone_name: str = 'Pearlfish'

@dataclass
class PearlfishW80MA(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k80MA
    model_config_name:str = 'PearlfishW80M'
    revise_backbone_name: str = 'Pearlfish'

@dataclass
class PearlfishW120MA(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k120MA
    model_config_name:str = 'PearlfishW120M'
    revise_backbone_name: str = 'Pearlfish'

@dataclass
class GoldfishW40MA(SlideConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Decay3k40MA
    model_config_name:str = 'GoldfishW40M'
    revise_backbone_name: str = 'Goldfish'


@dataclass
class SardinaW10M(SlideConfig):
    Embedding: MSFTanhTrendConfigV3
    Backbone:  Decay3k10M
    model_config_name:str = 'SardinaW10M'
    revise_backbone_name: str = 'Sardina'
    
@dataclass
class SardinaW40M(SlideConfig):
    Embedding: MSFTanhTrendConfigV3
    Backbone:  Decay3k40MA
    model_config_name:str = 'SardinaW40M'
    revise_backbone_name: str = 'Sardina'

@dataclass
class PearlfishW02M(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k02M
    model_config_name:str = 'PearlfishW02M'
    revise_backbone_name: str = 'Pearlfish'


@dataclass
class GoldfishW10M(SlideConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Decay3k10M
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'GoldfishW10M'


@dataclass
class PearlfishW10MC(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k10MC
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'PearlfishW10MC'
    revise_backbone_name: str = 'Pearlfish'

@dataclass
class PearlfishW20MB(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k20MB

    model_config_name:str = 'PearlfishW20MB'
    revise_backbone_name: str = 'Pearlfish'

@dataclass
class PearlfishW20M(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k20M
    model_config_name:str = 'PearlfishW20M'
    revise_backbone_name: str = 'Pearlfish'

@dataclass
class CrystalfishW02M(SlideConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay15h02M
    model_config_name:str = 'CrystalfishW02M'
    revise_backbone_name: str = 'Crystalfish'

######################################################
@dataclass
class RetNetSlidePyramidConfig(SlideConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Decay3k
    use_whole_layer_output: bool = True

    def __post_init__(self):
        assert self.use_whole_layer_output, "RetNetSlidePyramidConfig only support use_whole_layer_output=True"
        super().__post_init__()
        
    @property
    def signal_property(self):
        return 'SlidePyramid'

@dataclass
class PearlfishWP10MC(RetNetSlidePyramidConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k10MC
    Predictor: SlideWindowPredictorConfig
    model_config_name:str = 'PearlfishWP10MC'
    revise_backbone_name: str = 'Pearlfish'

############################################
########### Group End2End Model ############
############################################
@dataclass
class GroupE2ESlideConfig(SlideConfig):
    model_type: str = 'RetNetGroupSlide'
    Predictor: SlideWindowPredictorConfig
    use_flashattn_in_group_attention: bool = True
    only_train_multistation: bool = False
    @property
    def signal_property(self):
        return 'GroupE2E'
    def __post_init__(self):
        if not self.only_train_multistation:
            print(f"This setting `only_train_multistation=False` will train the single trace prediction branch, please double check.")
        return super().__post_init__()
@dataclass
class MultiFishW10M(GroupE2ESlideConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Decay3k10M
    model_config_name:str = 'MultiFishW10M'
    cross_num_hidden_layers: int = 1
    cross_hidden_size: int = 128
    group_decoder_class: str = 'DistanceAttention_Alpha'

@dataclass
class MultiFishW10M_Beta(MultiFishW10M):
    model_config_name:str = 'MultiFishW10M_V2'
    group_decoder_class: str = 'DistanceAttention_Beta'

@dataclass
class MultiFishW10M_Gamma(MultiFishW10M):
    model_config_name:str = 'MultiFishW10M_V3'
    group_decoder_class: str = 'DistanceAttention_Gamma'
    
@dataclass
class MultiFishW10M_Kappa(MultiFishW10M):
    model_config_name:str = 'MultiFishW10M_V4'
    group_decoder_class: str = 'DistanceAttention_Kappa'

@dataclass
class MultiFishW10M_Zeta(MultiFishW10M):
    model_config_name:str = 'MultiFishW10M_V5'
    group_decoder_class: str = 'DistanceAttention_Zeta'

@dataclass
class MultiFishW20M_Zeta(MultiFishW10M):
    model_config_name:str = 'MultiFishW20M_V5'
    Backbone:  Decay3k20M
    group_decoder_class: str = 'DistanceAttention_Zeta'

@dataclass
class MultiFishW40M(MultiFishW10M):
    model_config_name:str = 'MultiFishW40M_V5'
    Backbone:  Decay3k40MA
    group_decoder_class: str = 'DistanceAttention_Zeta'


############################################
########### Recurrent Model ################
############################################
@dataclass
class RecurrentConfig(SignalModelConfig):
    model_type: str = 'RetNetRecurrent'


@dataclass
class RetNetRecPhaseConfig(RecurrentConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Normal
    Predictor: RecurrentPredictorConfig
    model_config_name: str = 'RetNetRecurrent_MSF_Phase'

@dataclass
class RetNetRecReLUPhaseConfig(RecurrentConfig):
    Embedding: MSFReLUPhaseConfig
    Backbone:  Normal
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'RetNetRecurrent_ReLU_Phase'



@dataclass
class RetNetDecayReLUConfig(RecurrentConfig):
    Embedding: MSFReLUPhaseConfig
    Backbone:  Decay3k
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'RetNetDecay_ReLU'

@dataclass
class RetNetDecayConfig(RecurrentConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Decay3k
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'RetNetDecay'

@dataclass
class RetRecurrentNoBiasPhaseConfig(RecurrentConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  NoBias
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'RetRecurrent_NoBias_Phase'

@dataclass
class RetNetDecayNoBiasConfig(SignalModelConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Decay3kNoBias
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'RetNetDecay_NoBias'

@dataclass
class GoldfishR85M(RetNetDecayConfig):
    Backbone:  Decay3k85M
    model_config_name:str = 'GoldfishR85M'

@dataclass
class GoldfishR10MZ(RetNetDecayConfig):
    Backbone:  Decay3k10MZ
    model_config_name:str = 'GoldfishR10MZ'

@dataclass
class Goldfish10M(RetNetDecayConfig):
    Backbone:  Decay3k10M
    model_config_name:str = 'Goldfish10M'



############################################
########### SignalSea Model ################
############################################
@dataclass
class SignalSeaConfig(SignalModelConfig):
    model_type: str = 'RetNetSignalSea'
    scilence_alpha: int = field(default=1)
    retention_increment_type: str = field(default='kv_norm')
    scilence_loss_threshold: float = field(default=0)
    activate_loss_threshold: float = field(default=0)
    def __post_init__(self):
        super().__post_init__()
        if self.retention_increment_type == 'kv':
            print(f"""
                  WARNNING: you are directly use kv increment type, which is not optimized, please use qk increment type.
                  In math, it is same as retention_increment_type=='kv_norm'. Make sure you do want to use the increment.
                  Otherwise, I recommend use the new setting since 20231212
                  """)
            
    @property
    def signal_property(self):
        return 'Sea'

@dataclass
class DirectSeaConfig(SignalSeaConfig):
    model_type: str = 'RetNetDirctSea'
    Embedding: DFEConfig
    Backbone:  RetnetConfig
    Predictor: RecurrentPredictorConfig
    model_config_name: str = 'RetNetRecurrent_DirctlySea'
    @property
    def signal_property(self):
        return 'Directly.Sea'
    
@dataclass
class SilverfishS10M(DirectSeaConfig):
    Backbone:  Decay3k10M
    model_config_name:str = 'SilverfishS10M'

@dataclass
class RetNetRecSignalSeaConfig(SignalSeaConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Normal
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'RetRecurrent_SignalSea'

@dataclass
class RetRecDecaySignalSeaConfig(SignalSeaConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Decay3k
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'RetNetDecay_SignalSea'

@dataclass
class RetRecDecaySignalSeaReLUConfig(SignalSeaConfig):
    Embedding: MSFReLUPhaseConfig
    Backbone:  Decay3k
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'RetNetDecay_ReLU_SignalSea'
    @property
    def signal_property(self):
        return 'ReLU.Sea'
    
@dataclass
class ReLUfishS10M(RetRecDecaySignalSeaReLUConfig):
    Backbone:  Decay3k10M
    model_config_name:str = 'ReLUfishS10M'

@dataclass
class RetRecFishSignalSeaConfig(SignalSeaConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Decay1k
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'RetNetFish_SignalSea'

@dataclass
class GoldfishS85M(RetRecDecaySignalSeaConfig):
    Backbone:  Decay3k85M
    model_config_name:str = 'GoldfishS85M'

@dataclass
class GoldfishS40MA(RetRecDecaySignalSeaConfig):
    Backbone:  Decay3k40MA
    model_config_name:str = 'GoldfishS40M'

@dataclass
class FlatfishS40MA(RetRecDecaySignalSeaConfig):
    Backbone:  Decay1k40MA
    model_config_name:str = 'FlatfishS40M'




@dataclass
class PearlfishS40MA(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k40MA
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'PearlfishS40M'
    revise_backbone_name: str = 'Pearlfish'


@dataclass
class JadefishS40MA(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay75d40MA
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'JadefishS40M'
    revise_backbone_name: str = 'Jadefish'


@dataclass
class SardinaS40MA(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV3
    Backbone:  Decay3k40MA
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'SardinaS40M'
    revise_backbone_name: str = 'Sardina'
@dataclass
class SardinaS10MA(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV3
    Backbone:  Decay3k10M
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'SardinaS10M'
    revise_backbone_name: str = 'Sardina'
@dataclass
class SardinaReLUS10MA(SignalSeaConfig):
    Embedding: MSFReLUTrendConfigV3
    Backbone:  Decay3k10M
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'SardinaReLUS10M'
    revise_backbone_name: str = 'SardinaReLU'

@dataclass
class SardinaReLUS40MA(SignalSeaConfig):
    Embedding: MSFReLUTrendConfigV3
    Backbone:  Decay3k40MA
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'SardinaReLUS40M'
    revise_backbone_name: str = 'SardinaReLU'


@dataclass
class PearlfishS80MA(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k80MA
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'PearlfishS80M'
    revise_backbone_name: str = 'Pearlfish'


@dataclass
class GoldfishS85MB(RetRecDecaySignalSeaConfig):
    Backbone:  Decay3k85MB
    model_config_name:str = 'GoldfishS85MB'

@dataclass
class GoldfishS11MB(RetRecDecaySignalSeaConfig):
    Backbone:  Decay3k11MB
    model_config_name:str = 'GoldfishS11MB'


@dataclass
class GoldfishS10M(RetRecDecaySignalSeaConfig):
    Backbone:  Decay3k10M
    model_config_name:str = 'GoldfishS10M'

@dataclass
class GoldfishS20M(RetRecDecaySignalSeaConfig):
    Backbone:  Decay3k20M
    model_config_name:str = 'GoldfishS20M'

@dataclass
class PearlfishS20MB(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k20MB
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'PearlfishS20MB'
    revise_backbone_name: str = 'Pearlfish'


@dataclass
class FlatfishS10M(RetRecDecaySignalSeaConfig):
    Backbone:  Decay1k10M
    model_config_name:str = 'FlatfishS10M'

@dataclass
class FlatfishS02M(RetRecDecaySignalSeaConfig):
    Backbone:  Decay1k02M
    model_config_name:str = 'FlatfishS02M'


@dataclass
class PearlfishS10M(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k10M
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'PearlfishS10M'
    revise_backbone_name: str = 'Pearlfish'

@dataclass
class PearlfishS10ML1(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k10ML1
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'PearlfishS10ML1'
    revise_backbone_name: str = 'Pearlfish'


@dataclass
class PearlfishS02M(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k02M
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'PearlfishS02M'
    revise_backbone_name: str = 'Pearlfish'

@dataclass
class CrystalfishS10ML1(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay15h10ML1
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'CrystalfishS10ML1'
    revise_backbone_name: str = 'Crystalfish'


@dataclass
class CrystalfishS02M(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay15h02M
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'CrystalfishS02M'
    revise_backbone_name: str = 'Crystalfish'

@dataclass
class CrystalfishS10M(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay15h10M
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'CrystalfishS10M'
    revise_backbone_name: str = 'Crystalfish'

@dataclass
class PearlfishS10M(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k10M
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'PearlfishS10M'
    revise_backbone_name: str = 'Pearlfish'


@dataclass
class PearlfishS10MC(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k10MC
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'PearlfishS10MC'
    revise_backbone_name: str = 'Pearlfish'

@dataclass
class PearlfishS10MB(SignalSeaConfig):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k10MB
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'PearlfishS10MB'
    revise_backbone_name: str = 'Pearlfish'


@dataclass
class DiamondfishS10M(SignalSeaConfig):
    Embedding: MSFTanhTrendConfig
    Backbone:  Decay3k10M
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'DiamondfishS10M'
    revise_backbone_name: str = 'Diamondfish'



@dataclass
class GroupE2ESeaConfig(SignalSeaConfig):
    model_type: str = 'RetNetGroupSea'
    Predictor: RecurrentPredictorConfig
    use_flashattn_in_group_attention: bool = True
    only_train_multistation: bool = False
    @property
    def signal_property(self):
        return 'GroupE2E'
    def __post_init__(self):
        if not self.only_train_multistation:
            print(f"This setting `only_train_multistation=False` will train the single trace prediction branch, please double check.")
        return super().__post_init__()

@dataclass
class MultiFishS40M(GroupE2ESeaConfig):
    model_type: str = 'RetNetGroupSea'
    model_config_name:str = 'MultiFishS40M_V5'
    Embedding: MSFTanhPhaseConfig
    Backbone:  Decay3k40MA
    group_decoder_class: str = 'DistanceAttention_Zeta'
    cross_num_hidden_layers: int = 1
    cross_hidden_size: int = 128

############################################
########### SignalSeaL1 Model ##############
############################################
@dataclass
class SeaL1Config(SignalSeaConfig):
    model_type: str = 'RetNetSignalSeaL1'
    use_whole_layer_output: bool = True
    @property
    def signal_property(self):
        return 'L1'

@dataclass
class RetRecDecaySeaL1Config(SeaL1Config):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Decay3k
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'RetNetDecay_SignalSeaL1'

@dataclass
class GoldfishL10M(RetRecDecaySeaL1Config):
    Backbone:  Decay3k10M
    model_config_name:str = 'GoldfishL10M'

@dataclass
class GoldfishL85M(RetRecDecaySeaL1Config):
    Backbone:  Decay3k85M
    model_config_name:str = 'GoldfishL85M'

@dataclass
class GoldfishL85MB(RetRecDecaySeaL1Config):
    Backbone:  Decay3k85MB
    model_config_name:str = 'GoldfishL85MB'

@dataclass
class FlatfishL10M(RetRecDecaySeaL1Config):
    Backbone:  Decay1k10M
    model_config_name:str = 'FlatfishL10M'


@dataclass
class PearlfishL10M(SeaL1Config):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k10M
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'PearlfishL10M'
    revise_backbone_name: str = 'Pearlfish'

@dataclass
class PearlfishL40M(SeaL1Config):
    Embedding: MSFTanhTrendConfigV2
    Backbone:  Decay3k40MA
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'PearlfishL40M'
    revise_backbone_name: str = 'Pearlfish'



#### RetNetSignalLake
@dataclass
class SignalLakeConfig(SignalSeaConfig):
    model_type: str = 'RetNetSignalLake'
    output_k_score_in_column_space: str = field(
        default='normal', choices=['normal', 'detach']
        )
            
    @property
    def signal_property(self):
        return 'Lake'

@dataclass
class GoldfishZ10MZ(SignalLakeConfig):
    Embedding: MSFTanhPhaseConfig
    Backbone:  Decay3k10MZ
    Predictor: RecurrentPredictorConfig
    model_config_name:str = 'GoldfishZ10MZ'
