from dataclasses import dataclass
from typing import Optional, Union, List,Tuple
from simple_parsing import ArgumentParser, subgroups, field
import json
import numpy as np
import pandas as pd
import os

class HandleCreater:
    def __init__(self, handleclass, args, kargs):
        self.handleclass = handleclass
        self.args = args
        self.kargs = kargs
    def __call__(self):
        return self.handleclass(*self.args, **self.kargs)

###############################################
############## Resource Config  ###############
###############################################

@dataclass
class NoiseGenerateConfig:    
    noise_picking_rule: str
    noise_config_tracemapping_path: str = field(default=None)
    noise_config_noise_namepool: str = field(default=None)

    @property
    def name(self):
        raise NotImplementedError
    
class RandomPickAlongReceiveType:
    def __init__(self, df, trace_mapping):
        receiver_type_to_tracename_pool = df.groupby('receiver_type')['trace_name'].apply(lambda x:list(x))
        self.receiver_type_to_tracename_pool = receiver_type_to_tracename_pool
        self.trace_mapping = trace_mapping
        for must_key in ['HH', 'HN', 'EH']:
            assert must_key in receiver_type_to_tracename_pool, f"receiver_type_to_tracename_pool should contain {must_key} but not"
    def __getitem__(self, x, length=None):
        _type = self.trace_mapping[x]
        _type = 'HH' if _type not in self.receiver_type_to_tracename_pool else _type
        
        return np.random.choice(self.receiver_type_to_tracename_pool[_type])
    
    def get_trace_from_name(self, x, split='train'):
        _type = self.trace_mapping[x]
        _type = 'HH' if _type not in self.receiver_type_to_tracename_pool else _type
        if split.lower in ['train']:
            return np.random.choice(self.receiver_type_to_tracename_pool[_type])
        else:
            return self.receiver_type_to_tracename_pool[_type][0]

    def __repr__(self):
        string = [f"{key}->{len(val)}" for key,val in self.receiver_type_to_tracename_pool.items()]
        string = '\n'.join(string)
        return "RandomPickAlongReceiveType:\n"+string

@dataclass
class NoneNoiseGenerateConfig(NoiseGenerateConfig):
    noise_picking_rule: str = 'NoneNoiseGenerateConfig'
    def build_noise_engine(self):
        return None
    def __post_init__(self):
        assert not self.noise_config_noise_namepool , f"You should not assign noise_config_noise_namepool={self.noise_config_noise_namepool} for NoneNoiseGenerateConfig"
        assert not self.noise_config_tracemapping_path , f"You should not assign noise_config_tracemapping_path={self.noise_config_tracemapping_path} for NoneNoiseGenerateConfig"

    @property
    def name(self):
        return None

@dataclass
class PickNoiseAlongReceiveType_Config(NoiseGenerateConfig):
    noise_picking_rule: str = 'PickNoiseAlongReceiveType'
    def build_noise_engine(self):
        with open(self.noise_config_tracemapping_path, 'r') as f:
            trace_mapping = json.load(f)
        
        df = pd.read_csv(self.noise_config_noise_namepool)
        return {'train':RandomPickAlongReceiveType(df[df['split']=='TRAIN'], trace_mapping),
                'valid':RandomPickAlongReceiveType(df[df['split']=='DEV'], trace_mapping),
                'test':RandomPickAlongReceiveType(df[df['split']=='TEST'], trace_mapping)}
    def __post_init__(self):
        assert self.noise_config_noise_namepool is not None, "You should assign noise_config_noise_namepool for PickNoiseAlongReceiveType"
        assert self.noise_config_tracemapping_path is not None, "You should assign noise_config_tracemapping_path for PickNoiseAlongReceiveType"

    @property
    def name(self):
        return f"RevTypeNoise"
@dataclass
class ResourceConfig:
    resource_source: str = None
    load_in_all_processing: bool = field(default=False)
    bandfilter_rate: Union[float,str]= field(default=0)
    signal_lowpass: bool = field(default=False)
    downsample_rate: int = field(default=None)
    upsample_rate: int = field(default=None)
    use_resource_buffer: bool = field(default=False)
    initial_handle_inside_dataset: bool = field(default=False)
    NoiseGenerate: Optional[NoiseGenerateConfig] = subgroups(
        {
            "nonoise":NoneNoiseGenerateConfig,
            "pickalong_receive": PickNoiseAlongReceiveType_Config,
        }, default='nonoise'
    )
    amplifier_signal: int = field(default=0)

    def __post_init__(self):
        assert self.resource_source is not None, "It seem you are not assign one resource via --resource_source [resource] !!!! "
        if 'accu' in self.resource_source:
            assert self.amplifier_signal > 0, "when using the accelerate waveform, please manually set the amplifier, for accelerate, I recommend amplifier_signal=100. The performance diff may appear in low precision digit like bf16 and fp16"
    @property
    def resource_frequence(self):
        raise NotImplementedError("You should assign the resource_frequence for the resource")

    @property
    def resource_length(self):
        raise NotImplementedError("You should assign the resource_sampling_frequence for the resource")

    @property
    def sampling_frequence(self):
        if self.downsample_rate:
            assert self.upsample_rate is None, "You should not assign both downsample_rate and upsample_rate"
            return self.resource_frequence//self.downsample_rate
        elif self.upsample_rate:
            assert self.downsample_rate is None, "You should not assign both downsample_rate and upsample_rate"
            return self.resource_frequence*self.upsample_rate
        else:
            return self.resource_frequence

    @property
    def basemaxlength(self):
        if self.downsample_rate:
            assert self.upsample_rate is None, "You should not assign both downsample_rate and upsample_rate"
            return self.resource_length//self.downsample_rate
        elif self.upsample_rate:
            assert self.downsample_rate is None, "You should not assign both downsample_rate and upsample_rate"
            return self.resource_length*self.upsample_rate
        else:

            return self.resource_length
        
    @property
    def channel_order(self):
        raise NotImplementedError("You should assign the channel_order for the resource")

    @staticmethod
    def get_data_from_name(name):
        raise NotImplementedError("You should assign the get_data_from_name for the resource")
    
    @property
    def name(self):
        name = self.resource_source.split('/')[-1]
        name = name.replace('.hdf5','') #### <--- ensure the .hdf5 dataset is same as those without .hdf5
        if self.downsample_rate:
            assert self.upsample_rate is None, "You should not assign both downsample_rate and upsample_rate"
            name += f".DownsampleTo{self.sampling_frequence}Hz"
        elif self.upsample_rate:
            assert self.downsample_rate is None, "You should not assign both downsample_rate and upsample_rate"
            name += f".UpsampleTo{self.sampling_frequence}Hz"
        elif self.NoiseGenerate and self.NoiseGenerate.name is not None:
            name += f".add{self.NoiseGenerate.name}"
        

        return name
    


@dataclass
class ResourceSTEAD(ResourceConfig):
    bandfilter_rate: Union[float,str]= field(default=0)
    
    @property
    def resource_frequence(self):
        return 100
    
    @property
    def resource_length(self):
        return 6000

    @property
    def channel_order(self):
        return 'ENZ'
    
    def __post_init__(self):
        super().__post_init__()
        assert 'stead' in self.resource_source, "You should assign one STEAD resource !!!! or switch to other resource by --Resource"
        assert not self.signal_lowpass, "signal_lowpass is not implemented yet"
    @staticmethod
    def get_data_from_name(name):
        full_accelerate_f16 = [
                "datasets/STEAD/full_accelerate.f16.npy",
            ]
        full_accelerate_f32 = ["datasets/STEAD/full_accelerate.npy"]
        full_quantity_f16 = [
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.1.f16.npy",
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.2.f16.npy",
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.3.f16.npy",
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.4.f16.npy",
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.5.f16.npy",
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.6.f16.npy",
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.b7.f16.npy",
        ]
        full_quantity_f32 = [
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.1.npy",
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.2.npy",
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.3.npy",
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.4.npy",
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.5.npy",
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.6.npy",
            "datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.b7.npy",
        ]
        test_quantity_f32 = ["datasets/STEAD/LJSource/SplitFiles/stead.grouped.L.1.npy"]

        BDLEELSSO_DIR = "datasets/STEAD/BDLEELSSO/Split"
        BDLEELSSO_f32 = [os.path.join(BDLEELSSO_DIR,p) for p in os.listdir(BDLEELSSO_DIR) if 'index' not in p] if os.path.exists(BDLEELSSO_DIR) else None
        BDLEELSSO_trace = "datasets/STEAD/BDLEELSSO/trace.fast.csv"
        DEV_BDLEELSSO_trace  = "datasets/STEAD/BDLEELSSO/DEV_trace.fast.csv"
        DEV_BDLEELSSO_DIR = ["datasets/STEAD/BDLEELSSO/DEV_trace.waveform.npy"]

        quantity_means = "datasets/STEAD/LJSource/grouped.all.means.npy"
        quantity_stds  = "datasets/STEAD/LJSource/grouped.all.stds.npy"
        fast_trace_path = 'datasets/STEAD/stead.trace.fast.csv'
        all_trace_path  = 'datasets/STEAD/stead.base.csv'

        fast_small_trace_path = 'datasets/STEAD/stead.small.trace.fast.csv'
        stead_hdf5 = "datasets/STEAD/stead.hdf5"
        stead_buffer_hdf5_list = {"resource_type":"hdf5.path.list",
                                  "name2part": None, 
                                  "part_list":{0:"datasets/STEAD/stead.hdf5"},
                                  'dataflag':'stead'}
        ACCUDIR='datasets/STEAD/stead.accelerate/hdf5/'
        accu_name2part_path='datasets/STEAD/stead.accelerate/stead_accu.name2part.json'
        ACCUTRACE='datasets/STEAD/stead.accelerate/stead_accu.trace.fast.extend.csv'
        stead_accu_buffer_hdf5_list = {"resource_type":"hdf5.path.list",
                                        "name2part":accu_name2part_path, 
                                        "part_list":{int(os.path.basename(hdf5_part).replace('.hdf5','').split('_')[-1]):hdf5_part 
                                                     for hdf5_part in [os.path.join(ACCUDIR,p) for p in os.listdir(ACCUDIR) 
                                                                       if p.endswith('.hdf5')]},
                                        'dataflag':'stead'
                                        }

        name = name.split('_')[0]
        if name in ['magtime.stead.trace','Quake6000']:
            return fast_trace_path, full_accelerate_f16, None
        elif name == 'stead.trace.quantity':
            return fast_trace_path, full_quantity_f16, None
        elif name == 'stead.trace.normed_quantity':
            return fast_trace_path, full_quantity_f16, [quantity_means, quantity_stds]
        elif name == 'stead.trace.normed_quantityf32':
            return fast_trace_path, full_quantity_f32, [quantity_means, quantity_stds]
        elif name == 'stead.trace.test':
            return fast_trace_path, test_quantity_f32, [quantity_means, quantity_stds]
        elif name == 'stead.trace.normed_hdf5':
            return fast_trace_path, stead_hdf5, [quantity_means, quantity_stds]
        elif name == 'stead.trace.BDLEELSSO':
            return BDLEELSSO_trace, BDLEELSSO_f32, None
        elif name == 'stead.trace.BDLEELSSO.hdf5':
            return BDLEELSSO_trace, stead_buffer_hdf5_list, None ## stead_buffer_hdf5_list, None
        elif name == 'stead.accu.trace.full.extend.hdf5':
            return ACCUTRACE, stead_accu_buffer_hdf5_list, None ## stead_buffer_hdf5_list, None
        elif name == 'stead.trace.accu.part.hdf5':
            return ACCUTRACE, stead_buffer_hdf5_list, None
        elif name == 'stead.small.trace.hdf5':
            return fast_small_trace_path, stead_hdf5, None
        elif name == 'stead.full.trace.hdf5':
            return fast_trace_path, stead_buffer_hdf5_list, None
        elif name == 'DEV.stead.trace.BDLEELSSO':
            return DEV_BDLEELSSO_trace, DEV_BDLEELSSO_DIR, None
        elif name == 'stead.trace.full.extend.hdf5':
            return "datasets/STEAD/stead.trace.fast.extend.csv", stead_buffer_hdf5_list, None
        elif name == 'stead.trace.full.extend.test.hdf5':
                return "datasets/STEAD/stead.trace.fast.extend.csv", stead_hdf5, None
        else:
            raise NotImplementedError(f'Unknown dataset name: {name}')


@dataclass
class ResourceDiTing(ResourceConfig):
    bandfilter_rate: Union[float,str]= field(default=0.005)
    
    @property
    def resource_frequence(self):
        return 50
    
    @property
    def resource_length(self):
        return 9000

    @property
    def channel_order(self):
        return 'ZNE'
    
    def __post_init__(self):
        super().__post_init__()
        assert 'diting' in self.resource_source or 'seisT' in self.resource_source, "You should assign one DiTing resource !!!!  or switch to other resource by --Resource"
    
    @staticmethod
    def get_data_from_name(name):
        diting_DIR = "datasets/DiTing330km/split"
        diting_trace_path     = 'datasets/DiTing330km/diting.fast.csv'
        diting_half_trace_path = 'datasets/DiTing330km/diting.half.fast.csv'
        diting_mimic_trace_path = 'datasets/DiTing330km/diting.mimic.fast.csv'
        diting_small_trace_path = 'datasets/DiTing330km/diting.small.fast.csv'
        diting_tiny_trace_path = 'datasets/DiTing330km/diting.tiny.fast.csv'
        diting_name2part_path = 'datasets/DiTing330km/diting.name2part.json'
        diting_station_11_trace_path = 'datasets/DiTing330km/diting.station_11.fast.csv'
        diting_station_60_trace_path = 'datasets/DiTing330km/diting.station_60.fast.csv'
        diting_hdf5_path_list = {"resource_type":"hdf5.path.list",
                                 "name2part":diting_name2part_path, 
                                 "part_list":{int(os.path.basename(hdf5_part).replace('.hdf5','').split('_')[-1]):hdf5_part for hdf5_part in [os.path.join(diting_DIR,p) for p in os.listdir(diting_DIR) if (('.hdf5' in p) and ('_part_' in p))]},
                                 'dataflag':'diting'
                                 }
        name = name.split('_')[0]
        if name == 'diting.trace.hdf5':
            return diting_trace_path, diting_hdf5_path_list, None
        elif name == 'diting.lowerrerset.hdf5':
            return 'datasets/DiTing330km/diting.lowerrerset.csv', diting_hdf5_path_list, None
        elif name == 'diting.trace.500k.hdf5':
            return 'datasets/DiTing330km/diting.fast.M.csv', diting_hdf5_path_list, None
        elif name == 'diting.small.trace.hdf5':
            return diting_small_trace_path, diting_hdf5_path_list, None
        elif name == 'diting.tiny.trace.hdf5':
            return diting_tiny_trace_path, diting_hdf5_path_list, None
        elif name == 'diting.half.trace.hdf5':
            return diting_half_trace_path, diting_hdf5_path_list, None
        elif name == 'diting.mimic.trace.hdf5':
            return diting_mimic_trace_path, diting_hdf5_path_list, None
        elif name == 'diting.station11.trace.hdf5':
            return diting_station_11_trace_path, diting_hdf5_path_list, None
        elif name == 'diting.station60.trace.hdf5':
            return diting_station_60_trace_path, diting_hdf5_path_list, None
        elif name == 'seisT.trace.hdf5':
            return "datasets/DiTing330km/seisT.trace.csv", diting_hdf5_path_list, None
        elif name == 'seisT.group.hdf5':
            return "datasets/DiTing330km/seisT.group.csv", diting_hdf5_path_list, None
        elif name == 'ditinggroup.trace.hdf5':
            print(f"WARNING: {name}  is found not parted along the group correctly, please do not use it for group-like training anymore")
            return 'datasets/DiTing330km/ditinggroup.fast.csv', diting_hdf5_path_list, None
        elif name == 'ditinggroup.trace.xxs.hdf5':
            print(f"WARNING: {name}  is found not parted along the group correctly, please do not use it for group-like training anymore")
            return 'datasets/DiTing330km/ditinggroup.fast.xxs.csv', diting_hdf5_path_list, None
        
        elif name == 'ditinggroup.small.trace.hdf5':
            print(f"WARNING: {name} is found not parted along the group correctly, please do not use it for group-like training anymore")
            return 'datasets/DiTing330km/ditinggroupB3.small.fast.csv', diting_hdf5_path_list, None
        elif name == 'ditinggroup.trace.halfgood.hdf5':
            print(f"WARNING: {name}  is found not parted along the group correctly, please do not use it for group-like training anymore")
            return 'datasets/DiTing330km/ditinggroup.fast.halfgood.csv', diting_hdf5_path_list, None
        elif name == 'ditinggroup.full.trace.halfgood.hdf5':
            print(f"WARNING: {name} is found not parted along the group correctly, please do not use it for group-like training anymore")
            return 'datasets/DiTing330km/ditinggroup.full.halfgood.csv', diting_hdf5_path_list, None
        elif name == 'ditinggroup.they.cross.eval.hdf5':
            print(f"WARNING: {name} is found not parted along the group correctly, please do not use it for group-like training anymore")
            return 'datasets/DiTing330km/ditinggroup.they.cross_eval.csv', diting_hdf5_path_list, None
        elif name == 'ditinggroup.they.fast.cross.eval.hdf5':
            print(f"WARNING: {name} is found not parted along the group correctly, please do not use it for group-like training anymore")
            return 'datasets/DiTing330km/ditinggroup.they.fast.cross_eval.csv', diting_hdf5_path_list, None
        elif name == 'diting.they.trace.hdf5':
            print(f"WARNING: {name} is found not parted along the group correctly, please do not use it for group-like training anymore")
            return 'datasets/DiTing330km/diting.they.trace.csv', diting_hdf5_path_list, None
        elif name == 'diting.group.full.good.hdf5':
            return 'datasets/DiTing330km/ditinggroup.full.good.csv', diting_hdf5_path_list, None
        elif name == 'diting.group.full.good.L.hdf5':
            return 'datasets/DiTing330km/ditinggroup.full.good.l.csv', diting_hdf5_path_list, None
        elif name == 'diting.group.full.good.M.hdf5':
            return 'datasets/DiTing330km/ditinggroup.full.good.m.csv', diting_hdf5_path_list, None
        elif name == 'diting.group.full.good.S.hdf5':
            return 'datasets/DiTing330km/ditinggroup.full.good.s.csv', diting_hdf5_path_list, None
        elif name == 'diting.group.full.good.XS.hdf5':
            return 'datasets/DiTing330km/ditinggroup.full.good.xs.csv', diting_hdf5_path_list, None
        elif name == 'diting.group.full.good.XXS.hdf5':
            return 'datasets/DiTing330km/ditinggroup.full.good.xxs.csv', diting_hdf5_path_list, None
        elif name == 'diting.group.good.hdf5':
            return 'datasets/DiTing330km/ditinggroup.fast.good.csv', diting_hdf5_path_list, None
        elif name == 'ditinggroup.full.subcluster.hdf5':
            return 'datasets/DiTing330km/ditinggroup.full.subcluster.csv', diting_hdf5_path_list, None
        else:
            raise NotImplementedError(f'Unknown dataset name: {name}')


@dataclass
class ResourceInstance(ResourceConfig):
    bandfilter_rate: Union[float,str]= field(default=0)
    
    @property
    def resource_frequence(self):
        return 100
    
    @property
    def resource_length(self):
        return 12000

    @property
    def channel_order(self):
        return 'ENZ'
    
    def __post_init__(self):
        super().__post_init__()
        assert 'instance' in self.resource_source , "You should assign one Instance resource !!!!  or switch to other resource by --Resource"
    
    @staticmethod
    def get_data_from_name(name):
        instance_ground_motion_hdf5 = "datasets/INSTANCE/Instance_events_gm.hdf5"
        instance_ground_motion_path_list = {"resource_type":"hdf5.path.list",
                                 "name2part": None, 
                                 "part_list":{0:"datasets/INSTANCE/Instance_events_gm.hdf5"},
                                 'dataflag':'instance'}
        
        name = name.split('_')[0]
        if name == 'instance.group.EH.hdf5':
            return "datasets/INSTANCE/instance.group.EH.csv", instance_ground_motion_path_list, None
        elif name == 'instance.group.HH.hdf5':
            return "datasets/INSTANCE/instance.group.HH.csv", instance_ground_motion_path_list, None
        elif name == 'instance.group.HN.hdf5':
            return "datasets/INSTANCE/instance.group.HN.csv", instance_ground_motion_path_list, None
        elif name == 'instance.group.all.hdf5':
            return "datasets/INSTANCE/instance.group.fast.csv", instance_ground_motion_path_list, None
        else:
            raise NotImplementedError(f'Unknown dataset name: {name}')


@dataclass
class ResourceCompound(ResourceConfig):
    bandfilter_rate: Union[float,str]= field(default=0)
    
    @property
    def resource_frequence(self):
        return 100
    
    @property
    def resource_length(self):
        return 6000

    @property
    def channel_order(self):
        return 'ENZ'
    
    def __post_init__(self):
        super().__post_init__()
        assert 'compound' in self.resource_source, "You should assign one Compound resource !!!! or switch to other resource by --Resource"
        assert not self.signal_lowpass, "signal_lowpass is not implemented yet"
    @staticmethod
    def get_data_from_name(name):
        compound_DIR = "datasets/Compound/STEAD+INSTANCE/split"
        STEADACCUDIR= 'datasets/STEAD/stead.accelerate/hdf5'
        INSTANCEDIR = 'datasets/INSTANCE/waveform_accelerate/split_hdf5'
        stead_and_instance_path_list = {"resource_type":"hdf5.path.list",
                                        "name2part":"datasets/Compound/STEAD+INSTANCE/name2part.json", 
                                        "part_list":{0:"datasets/STEAD/stead.hdf5",1:"datasets/INSTANCE/Instance_events_gm.hdf5"},
                                        'dataflag':'compound'
                                        }
        stead_and_instance_path_ceph = {
                                 "resource_type":"hdf5.path.list",
                                 "name2part":"datasets/Compound/STEAD+INSTANCE/name2part.json", 
                                 "part_list":{0:"s3://earthquake/STEAD/merge.hdf5",
                                              1:"s3://earthquake/INSTANCE/Instance_events_gm.hdf5"},
                                 'dataflag':'compound'
                                 }
        stead_and_instance_trace_csv = "datasets/Compound/STEAD+INSTANCE/compound.trace.fast.csv"
        stead_and_instance_trace_acc_csv = "datasets/Compound/STEAD+INSTANCE/compound.trace.accu.fast.csv"
        
        stead_and_instance_accu = {"resource_type":"hdf5.path.list",
                                    "name2part":"datasets/Compound/STEAD+INSTANCE/accu.name2part.json", 
                                    "part_list": {   f"stead_{int(os.path.basename(hdf5_part).replace('.hdf5',''))}":hdf5_part  for hdf5_part in [os.path.join(STEADACCUDIR,p) for p in os.listdir(STEADACCUDIR)  if p.endswith('.hdf5')]
                                               }|{f"instance_{int(os.path.basename(hdf5_part).replace('.hdf5',''))}":hdf5_part  for hdf5_part in [os.path.join(INSTANCEDIR, p) for p in os.listdir(INSTANCEDIR )  if p.endswith('.hdf5')]
                                               },
                                    'dataflag':'compound'
                                    }
        print(stead_and_instance_accu)
        name = name.split('_')[0]
        if name == 'compound.stead+instance.hdf5':
            return stead_and_instance_trace_csv, stead_and_instance_path_list, None
        elif name == 'compound.stead+instance.ceph':
            return stead_and_instance_trace_csv, stead_and_instance_path_ceph, None
        elif name == 'compound.stead+instance.accu.hdf5':
            return stead_and_instance_trace_acc_csv, stead_and_instance_accu, None
        else:
            raise NotImplementedError
        
class ResourceGroup:
    @staticmethod
    def find_group_key(df):
        possible_keys = df.keys()
        alled_keys = ['ev_id','group']
        if 'ev_id' in possible_keys:
            assert 'group' not in possible_keys, "ev_id and group cannot be in the same dataframe"
            return 'ev_id'
        elif 'group' in possible_keys:
            assert 'ev_id' not in possible_keys, "ev_id and group cannot be in the same dataframe"
            return 'group'
        else:
            raise Exception(f"cannot find the group key like {alled_keys}, now keys are {possible_keys}")


        
class ResourceGroup:
    @staticmethod
    def find_group_key(df):
        possible_keys = df.keys()
        alled_keys = ['ev_id','group']
        if 'ev_id' in possible_keys:
            assert 'group' not in possible_keys, "ev_id and group cannot be in the same dataframe"
            return 'ev_id'
        elif 'group' in possible_keys:
            assert 'ev_id' not in possible_keys, "ev_id and group cannot be in the same dataframe"
            return 'group'
        else:
            raise Exception(f"cannot find the group key like {alled_keys}, now keys are {possible_keys}")


@dataclass
class ResourceDiTingGroup(ResourceDiTing,ResourceGroup):
    
    def __post_init__(self):
        super().__post_init__()
        assert 'group' in self.resource_source, "You should assign one ditinggroup resource !!!!  or switch to other resource by --Resource"
    


    @staticmethod
    def get_data_from_name(name):
        diting_DIR              = "datasets/DiTing330km/split"
        diting_trace_path       = 'datasets/DiTing330km/ditinggroup.fast.csv'
        diting_group_path       = 'datasets/DiTing330km/ditinggroup.fast.good.csv'
        diting_small_trace_path = 'datasets/DiTing330km/ditinggroupB3.small.fast.csv'
        diting_tiny_trace_path  = 'datasets/DiTing330km/ditinggroupB3.tiny.fast.csv'
        diting_name2part_path   = 'datasets/DiTing330km/diting.name2part.json'
        diting_hdf5_path_list = {"resource_type":"hdf5.path.list",
                                "name2part":diting_name2part_path, 
                                "part_list":{int(os.path.basename(hdf5_part).replace('.hdf5','').split('_')[-1]):hdf5_part for hdf5_part in [os.path.join(diting_DIR,p) for p in os.listdir(diting_DIR) if (('.hdf5' in p) and ('_part_' in p))]}
                                }
        name = name.split('_')[0]
        if name == 'ditinggroup.trace.hdf5':
            raise NotImplementedError("ditinggroup.trace.hdf5 is found not parted along the group correctly, please use diting.group.good.hdf5")
            return diting_trace_path, diting_hdf5_path_list, None
        elif name == 'ditinggroup.small.trace.hdf5':
            #raise NotImplementedError("ditinggroup.trace.hdf5 is found not parted along the group correctly, please use diting.group.good.hdf5")
            return diting_small_trace_path, diting_hdf5_path_list, None
        elif name == 'ditinggroup.tiny.trace.hdf5':
            raise NotImplementedError("ditinggroup.trace.hdf5 is found not parted along the group correctly, please use diting.group.good.hdf5")
            return diting_tiny_trace_path, diting_hdf5_path_list, None
        elif name == 'diting.group.full.good.hdf5':
            return 'datasets/DiTing330km/ditinggroup.full.good.csv', diting_hdf5_path_list, None
        elif name == 'diting.group.good.hdf5':
            return 'datasets/DiTing330km/ditinggroup.fast.good.csv', diting_hdf5_path_list, None
        elif name == 'diting.group.full.good.XXS.hdf5':
            return 'datasets/DiTing330km/ditinggroup.full.good.xxs.csv', diting_hdf5_path_list, None
        elif name == 'diting.group.full.good.M.hdf5':
            return 'datasets/DiTing330km/ditinggroup.full.good.m.csv', diting_hdf5_path_list, None
        else:
            raise NotImplementedError(f'Unknown dataset name: {name}')

@dataclass
class ResourceInstanceGroup(ResourceInstance,ResourceGroup):
    
    def __post_init__(self):
        super().__post_init__()
        assert 'instance.group' in self.resource_source, "You should assign one instance.group resource !!!!  or switch to other resource by --Resource"
    
    @staticmethod
    def get_data_from_name(name):
        instance_ground_motion_hdf5 = "datasets/INSTANCE/Instance_events_gm.hdf5"
        instance_ground_motion_path_list = {"resource_type":"hdf5.path.list",
                                 "name2part": None, 
                                 "part_list":{0:"datasets/INSTANCE/Instance_events_gm.hdf5"},
                                 'dataflag':'instance'}
        
        name = name.split('_')[0]
        if name == 'instance.group.EH.hdf5':
            return "datasets/INSTANCE/instance.group.EH.csv", instance_ground_motion_path_list, None
        elif name == 'instance.group.HH.hdf5':
            return "datasets/INSTANCE/instance.group.HH.csv", instance_ground_motion_path_list, None
        elif name == 'instance.group.HN.hdf5':
            return "datasets/INSTANCE/instance.group.HN.csv", instance_ground_motion_path_list, None
        elif name == 'instance.group.all.hdf5':
            return "datasets/INSTANCE/instance.group.fast.csv", instance_ground_motion_path_list, None
        else:
            raise NotImplementedError(f'Unknown dataset name: {name}')





@dataclass
class MultiStationResourceConfig():
    data_root_dir: str = field(default='datasets/')
    dataset_name: str = field(default='STEAD')
    generate_offline_metadata: str = field(default=None)
    offline_metadata_path: str = field(default=None)
    metadata_keys: List[str] = field(default=None)
    waveform_type: str = field(default='count')
    filter_waveform: bool = field(default=False)    
    station_channel: Optional[List[str]] = field(default=None)
    waveform_path: Optional[str] = field(default=None)
    trace_category: Optional[str] = field(default=None)
    target_type: Optional[str] = field(default=None)
    window_length: Optional[int] = field(default=None)
    waveform_type: Optional[str] = field(default='count')
    numpy_dtypye: Optional[str] = field(default='float16')
    delta_xyz: Optional[bool] = field(default=None)
    align_time: Optional[bool] = field(default=None)
    split: str = field(default='TRAIN')
    data_type: str = field(default='numpy')
    split_file: str = field(default=None)
    
    def save_json(self, path: str):
        with open(path, "w") as f:
            config_dict = asdict(self)
            json.dump(config_dict, f, indent=1)
    
    @staticmethod
    def load_json(path: str):
        with open(path, "r") as f:
            config_dict = json.load(f)
        return ResourceConfig(**config_dict)
    
    

    
    
    
    def get_name(self):
        raise NotImplementedError
    
    @property
    def dataset_type(self):
        raise NotImplementedError
