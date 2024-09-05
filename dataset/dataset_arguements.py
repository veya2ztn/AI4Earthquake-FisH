from dataclasses import dataclass
from typing import Optional, Union, List,Tuple
from simple_parsing import ArgumentParser, subgroups, field
from .resource.resource_arguements import *
 

@dataclass
class SamplingStrategy:
    strategy_name: str = field(default="random_sample_in_ahead_L_to_p")
    early_warning: int = field(default=None)
    #sampling_range_based_on_p: Tuple[int, int] = (None, 0)

@dataclass
class EarlyWarningStrategy(SamplingStrategy):
    strategy_name: str = "early_warning_before_p"
    early_warning: int = 200



@dataclass
class ProjectSamplingStrategy:
    train_sampling_strategy: SamplingStrategy
    valid_sampling_strategy: EarlyWarningStrategy
    test_sampling_strategy:  EarlyWarningStrategy
# it seem accelerate cannot hanlde 

###############################################
############## Dataset Config  ################
###############################################
@dataclass
class DatasetConfig:
    Resource: ResourceConfig = subgroups(
        {
            "STEAD": ResourceSTEAD,
            "DiTing": ResourceDiTing,
            "DiTingGroup": ResourceDiTingGroup,
            "Instance": ResourceInstance,
            "Compound": ResourceCompound,
            "InstanceGroup": ResourceInstanceGroup,
        }, default="STEAD"
    )   
    debug: bool = field(default=False)
    dataset_name : str = field(default=None)
    load_in_all_processing: bool = field(default=False)
    return_trend: bool = field(default=False)   
    dataset_version: str = field(default='alpha')
    @property
    def name(self):
        if self.debug:return "debug"
        raise NotImplementedError
    
@dataclass
class QuakeDatasetConfig(DatasetConfig):
    max_length: int = field(default=3000)
    peak_relaxtion: int = field(default=None)

    def __post_init__(self):
    
        self.dataset_name = self.Resource.name

    @property
    def name(self):
        if self.debug:return "debug"
        return self.dataset_name
    
@dataclass 
class MultistationDatasetConfig(DatasetConfig):
    share_memory: bool = field(default=False, help='share_memory_flag')
    adjust_mean: str = field(default=None)
    trigger_based: str = field(default=None)
    fake_borehole: str = field(default=None)
    shuffle: str = field(default=None)
    windowlen: int = field(default=None)
    cutout: list = field(default=None)
    max_stations: int = field(default=None)
    shard_size: int = field(default=None)
    oversample: int = field(default=None)
    max_upsample_magnitude: int = field(default=None)
    min_upsample_magnitude: int = field(default=None)
    magnitude_resampling: int = field(default=None)
    upsample_high_station_events: bool = field(default=None)
    integrate: bool = field(default=None)
    
    
    def get_name(self):
        raise NotImplementedError
    
    @property
    def dataset_type(self):
        raise NotImplementedError
    
@dataclass
class TraceDatasetConfig(QuakeDatasetConfig):
    warning_window : int = field(default=None)
    status_type : str= field(default='N0P1S2')
    slide_stride: int = field(default=None)
    #return_idx: bool = field(default=False) # <-- this is not control here
    remove_noise: bool = field(default=False)
    use_db_waveform: bool = field(default=False)
    
    @property
    def dataclass_type(self):
        return "EarthQuakePerTrack" ## it is quite wired that I can not use dataclass assign the dataclass_type

@dataclass
class ConCatDatasetConfig(TraceDatasetConfig):
    #component_num_to_concat_longterm: int = field(default=2)
    component_concat_file: str = field(default=None)
    component_intervel_length: int = field(default=3000)
    #use_zero_padding_in_concat: bool = field(default=False)
    padding_rule: str = field(default="interpolation",choice=["interpolation","zero","repeat","noise"])

    @property
    def dataclass_type(self):
        return "ConCatDataset" ## it is quite wired that I can not use dataclass assign the dataclass_type
    # @property
    # def use_zero_padding_in_concat(self):
    #     if self.use_zero_padding_in_concat and self.padding_rule == "zero_padding":
    #         return True
        

    @property
    def name(self):
        if self.debug:return "debug"
        return self.dataset_name + f".ContinuesQuake"
    
class GroupDatasetConfig(TraceDatasetConfig):
    max_station : int = 8
    @property
    def dataclass_type(self):
        return "EarthQuakePerGroup" ## it is quite wired that I can not use dataclass assign the dataclass_type
###############################################
############## Dataloader Config  #############
###############################################
@dataclass
class DataloaderConfig:
    Dataset : DatasetConfig= subgroups(
        {
            "TraceDataset": TraceDatasetConfig,
            "ConCatDataset": ConCatDatasetConfig,
            "GroupDataset": GroupDatasetConfig,
        }, default="TraceDataset"
    )   
    shuffle : bool = field(default=True)
    num_workers : int= field(default=0)
    batch_size : int = field(default=2)
    data_parallel_dispatch: bool = field(default=False)
    donot_use_accelerate_dataloader: bool = field(default=False)
    not_pin_memory: bool = field(default=False)
    loader_all_data_in_memory_once: bool = field(default=False)
    def __post_init__(self):
        self.Dataset.load_in_all_processing = self.data_parallel_dispatch
        self.Dataset.Resource.load_in_all_processing = self.data_parallel_dispatch


