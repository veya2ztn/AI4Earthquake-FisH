

from dataclasses import dataclass
from typing import Optional, Union, List,Tuple
from simple_parsing import ArgumentParser, subgroups, field
from .optimizer.optimizer_arguements import OptimizerConfig
from .scheduler.scheduler_arguements import SchedulerConfig,TimmCosineConfig, NoSchedulerConfig, TransformersCosineConfig
from dataset.dataset_arguements import ProjectSamplingStrategy
from model.model_arguements import FreezeConfig
###############################################
############## Dataset Config  ################
###############################################
import os
from pathlib import Path
@dataclass
class MonitorConfig:
    use_wandb: bool = field(default=False)
    wandbwatch: bool = field(default=False)
    log_interval: int = field(default=50)


@dataclass
class CheckpointConfig:
    num_max_checkpoints: int = field(default=1)
    save_every_epoch: int = field(default=1)
    save_warm_up: int = field(default=5)
    preload_weight: Optional[str] = None
    preload_state: Optional[str] = None
    load_weight_partial: bool = field(default=False)
    load_weight_ignore_shape: bool = field(default=False)
    
@dataclass
class TaskConfig:
    pass

@dataclass
class TrainConfig(TaskConfig):
    Optimizer: OptimizerConfig
    Monitor: MonitorConfig
    Checkpoint: CheckpointConfig
    sampling_strategy:ProjectSamplingStrategy
    Freeze      : FreezeConfig
    Scheduler: SchedulerConfig = subgroups(
        {
            "TimmCosine": TimmCosineConfig,
            "none": NoSchedulerConfig,
            "TFCosine": TransformersCosineConfig,
        },
        default='TimmCosine'
    )
    
    clean_checkpoints_at_end: bool = field(default=False)
    epochs: int = field(default=100)
    seed: int = field(default=42)
    gradient_accumulation_steps : int = field(default=1)
    clip_value: Optional[float] = None
    not_valid_during_train : bool = field(default=False)
    lr : float = field(default=1e-5)
    continue_train: bool = field(default=False)
    find_unused_parameters: bool = field(default=False)
    save_on_epoch_end: bool = field(default=True)
    train_or_infer: str = field(default='train')
    do_validation_at_first_epoch: bool = field(default=False)
    autoresume: Optional[str] = None
    
    time_test: bool = field(default=False)
    preduce_meanlevel: str = field(default=None)
    def __post_init__(self):
        self.Optimizer.lr = self.lr
        self.Scheduler.lr = self.lr
        self.Scheduler.epochs = self.epochs
        self.train_or_infer = 'train'
        if self.autoresume:
            ### we will auto search the latest checkpoint
            checkpoints_root = Path(os.path.join(self.autoresume, 'checkpoints'))
            checkpoints_dirs = list(checkpoints_root.glob("checkpoint_*"))
            if len(checkpoints_dirs)>0:
                print(f"auto resume from {checkpoints_dirs[0]}")
                self.Checkpoint.preload_weight = ""
                self.Checkpoint.preload_state  = checkpoints_dirs[0]
                self.continue_train = True
     
