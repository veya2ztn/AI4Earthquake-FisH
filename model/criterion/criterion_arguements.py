from dataclasses import dataclass
from typing import Optional, Union, List, Tuple
from simple_parsing import field


@dataclass
class LossConfig:
    pass

@dataclass
class PosslossConfig:
    final_sequence_length:int = None
    inc_var_loss:bool = True
    

    
     