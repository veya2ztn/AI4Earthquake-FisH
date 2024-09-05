from dataclasses import dataclass
from typing import Optional, Union, List,Tuple
from simple_parsing import ArgumentParser, subgroups, field
from train.train_arguements import TaskConfig, CheckpointConfig
from dataset.dataset_arguements import ProjectSamplingStrategy, EarlyWarningStrategy



# @dataclass
# class InferSamplingStrategy:
    
#     valid_sampling_strategy: EarlyWarningStrategy

@dataclass
class EvaluatorConfig(TaskConfig):
    Checkpoint       : CheckpointConfig
    sampling_strategy: ProjectSamplingStrategy
    infer_mode       : str = None
    upload_to_wandb  : bool = False
    clean_up_plotdata: bool = False
    train_or_infer : str = 'infer'
    calculate_the_correlation: bool = False
    eval_dataflag: str = 'DEV'
    save_monitor_num: int = 10
    do_fasttest_eval: bool = False
    def __post_init__(self):
        self.train_or_infer   = 'infer'
        
@dataclass
class NormalInferenceMode(EvaluatorConfig):
    infer_mode       : str = 'normal'

@dataclass
class RecurrentInferenceMode(EvaluatorConfig):
    infer_mode: str = 'recurrent'
    recurrent_chunk_size: int = field(default=3000)
    recurrent_start_size: int = field(default=3000)
    recurrent_slide_output_strider: int = field(default=1)
@dataclass
class SlideInferenceMode(EvaluatorConfig):
    infer_mode: str   = 'slide'
    slide_stride: int = field(default=100)

@dataclass
class EvalPlotConfig(TaskConfig):
    sampling_strategy: ProjectSamplingStrategy
    plot_data_dir: str 
    infer_mode    : str = None
    upload_to_wandb: bool = False
    train_or_infer : str = 'infer_plot'
    calculate_the_correlation: bool = False
    recurrent_slide_output_strider: int = field(default=1)
    do_fasttest_eval: bool = False
    def __post_init__(self):
        assert self.infer_mode is not None,  "infer_mode must be specified"
        self.train_or_infer = 'infer_plot'