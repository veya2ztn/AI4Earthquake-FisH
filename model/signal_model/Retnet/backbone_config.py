from ..backbone_config import BackboneConfig
from dataclasses import dataclass
from simple_parsing import field
from config.utils import get_compare_namespace_trees, get_default_config


@dataclass
class RetnetConfig(BackboneConfig):
    architecture: str = "RetNet"
    vocab_size :int = 3 #<--------we dont use it, only to keep the same name as previous
    disable_all_bias: bool = False
    use_lm_decay: int = None
    
    normlize_for_stable: int  = field(default=-1)
    use_flash_retention: bool = field(default=False)
    normalize_at_end: bool = field(default=True)

    preset_architecture = False #<---must be False, otherwise may goes into infinity loop when call type(self)()
    retention_mode: str = field(default='qk_first')
    retention_dropout : float = field(default=0.0)
    monitor_retention: bool = field(default=False)
    groupnorm_type: str = field(default='RMSNorm')
    backbone_type = None
    #normalize_before_retention: bool = field(default=False)
    
    def __post_init__(self):
        super().__post_init__()
        if not self.preset_architecture: return
        for key in ['use_lm_decay','disable_all_bias']:
            assert getattr(self, key) == getattr(type(self), key), f"""
            For fixed backbone config, you are now allowed custom the backbone config, below is the compare result:\n{get_compare_namespace_trees(self, type(self)())}
            """

@dataclass
class PresetRetnetConfig(RetnetConfig):
    preset_architecture = True
    backbone_type = None
    
@dataclass
class Normal(PresetRetnetConfig):
    use_lm_decay: int = None
    backbone_type = "Zebrafish"
    

@dataclass
class Decay3k(PresetRetnetConfig):
    use_lm_decay: int = 3000
    backbone_type = "Goldfish"

@dataclass
class Decay15h(PresetRetnetConfig):
    use_lm_decay: int = 1500
    backbone_type = "Yellowfish"

@dataclass
class Decay75d(PresetRetnetConfig):
    use_lm_decay: int = 750
    backbone_type = "Redfish"

@dataclass
class Decay1k(PresetRetnetConfig):
    use_lm_decay: int = 1000
    backbone_type = "Flatfish"


@dataclass
class NoBias(Normal):
    disable_all_bias: bool = True

@dataclass
class Decay3kNoBias(Decay3k):
    disable_all_bias: bool = True

@dataclass
class Model10MZ(BackboneConfig):
    hidden_size:       int = 768
    intermediate_size: int = 512
    num_hidden_layers: int = 4
    attention_hidden_size: int = 256
    num_heads:int =  16
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "10M_Z"

@dataclass
class Model85M(BackboneConfig):
    hidden_size          : int = 1024
    intermediate_size    : int = 2048
    num_hidden_layers    : int = 8
    attention_hidden_size: int = 1024
    num_heads            : int = 16
    preset_model= True #<---- must has type assign like :bool

    @property
    def size_type(self):
        return  "85M_A"

@dataclass
class Model85MB(BackboneConfig):
    hidden_size          : int = 1024
    intermediate_size    : int = 2048
    num_hidden_layers    : int = 8
    attention_hidden_size: int = 1024
    num_heads            : int = 32
    preset_model= True #<---- must has type assign like :bool

    @property
    def size_type(self):
        return  "85M_B"

@dataclass
class Model11MB(BackboneConfig):
    hidden_size          : int = 1024
    intermediate_size    : int = 2048
    num_hidden_layers    : int = 1 #<--Notice here, it is the first layer of 85MB model
    attention_hidden_size: int = 1024
    num_heads            : int = 32
    preset_model= True #<---- must has type assign like :bool

    @property
    def size_type(self):
        return  "11M_B"

@dataclass
class Model10M(BackboneConfig):
    hidden_size:       int = 512
    intermediate_size: int = 1024
    num_hidden_layers: int = 4
    attention_hidden_size: int = 256
    num_heads:int =  16
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "10M"

@dataclass
class Model10MB(BackboneConfig):
    hidden_size:       int = 512
    intermediate_size: int = 1024
    num_hidden_layers: int = 4
    attention_hidden_size: int = 256
    num_heads:int =  32
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "10MB"


@dataclass
class Model10MC(BackboneConfig):
    hidden_size:       int = 512
    intermediate_size: int = 1024
    num_hidden_layers: int = 4
    attention_hidden_size: int = 256
    num_heads:int =  64
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "10MC"


@dataclass
class Model20M(BackboneConfig):
    hidden_size:       int = 512
    intermediate_size: int = 1024
    num_hidden_layers: int = 8
    attention_hidden_size: int = 256
    num_heads:int =  16
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "20M"

@dataclass
class Model20MB(BackboneConfig):
    hidden_size:       int = 768
    intermediate_size: int = 1536
    num_hidden_layers: int = 4
    attention_hidden_size: int = 256
    num_heads:int =  16
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "20MB"

@dataclass
class Model20MA(BackboneConfig):
    hidden_size          : int = 1024
    intermediate_size    : int = 2048
    num_hidden_layers    : int = 2
    attention_hidden_size: int = 768
    num_heads            : int = 16
    preset_model= True #<---- must has type assign like :bool

    @property
    def size_type(self):
        return  "20M_A"

@dataclass
class Model40MA(BackboneConfig):
    hidden_size          : int = 1024
    intermediate_size    : int = 2048
    num_hidden_layers    : int = 4
    attention_hidden_size: int = 768
    num_heads            : int = 16
    preset_model= True #<---- must has type assign like :bool

    @property
    def size_type(self):
        return  "40M_A"

@dataclass
class Model10ML1(BackboneConfig):
    hidden_size          : int = 1024
    intermediate_size    : int = 2048
    num_hidden_layers    : int = 1
    attention_hidden_size: int = 768
    num_heads            : int = 16
    preset_model= True #<---- must has type assign like :bool

    @property
    def size_type(self):
        return  "10M_L1"

@dataclass
class Model80MA(BackboneConfig):
    hidden_size          : int = 1024
    intermediate_size    : int = 2048
    num_hidden_layers    : int = 8
    attention_hidden_size: int = 768
    num_heads            : int = 16
    preset_model= True #<---- must has type assign like :bool

    @property
    def size_type(self):
        return  "80M_A"


@dataclass
class Model120MA(BackboneConfig):
    hidden_size          : int = 1024
    intermediate_size    : int = 2048
    num_hidden_layers    : int = 12
    attention_hidden_size: int = 768
    num_heads            : int = 16
    preset_model= True #<---- must has type assign like :bool

    @property
    def size_type(self):
        return  "120M_A"

@dataclass
class Model02M(BackboneConfig):
    hidden_size:       int = 512
    intermediate_size: int = 1024
    num_hidden_layers: int = 1
    attention_hidden_size: int = 256
    num_heads:int =  16
    preset_model= True #<---- must has type assign like :bool
    
    @property
    def size_type(self):
        return "02M"
### put the preset config at first

@dataclass
class Decay3k02M(Model02M,Decay3k): pass


@dataclass
class Decay3k10M(Model10M,Decay3k):pass

@dataclass
class Decay3k10MB(Model10MB,Decay3k):pass

@dataclass
class Decay3k10MC(Model10MC,Decay3k):pass

@dataclass
class Decay3k11MB(Model11MB,Decay3k):pass


@dataclass
class Decay3k10MZ(Model10MZ,Decay3k):pass

@dataclass
class Decay1k02M(Model02M,Decay1k):pass



@dataclass
class Decay1k10M(Model10M,Decay1k):pass

@dataclass
class Decay15h10M(Model10M,Decay15h):pass

@dataclass
class Decay75d10M(Model10M,Decay75d):pass

@dataclass
class Decay75d40MA(Model40MA,Decay75d):pass

@dataclass
class Decay15h02M(Model02M,Decay15h):pass

@dataclass
class Decay15h10ML1(Model10ML1,Decay15h):pass

@dataclass
class Decay3k10ML1(Model10ML1,Decay3k):pass

@dataclass
class Decay3k85M(Model85M,Decay3k): pass

@dataclass
class Decay3k20M(Model20M,Decay3k): pass

@dataclass
class Decay3k20MB(Model20MB,Decay3k): pass

@dataclass
class Decay3k85MB(Model85MB,Decay3k): pass

@dataclass
class Decay3k40MA(Model40MA,Decay3k): pass

@dataclass
class Decay3k20MA(Model20MA,Decay3k): pass


@dataclass
class Decay1k40MA(Model40MA,Decay1k):pass

@dataclass
class Decay3k80MA(Model80MA,Decay3k): pass

@dataclass
class Decay3k120MA(Model120MA,Decay3k): pass
