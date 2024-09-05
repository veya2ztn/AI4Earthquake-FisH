from simple_parsing import ArgumentParser
from dataclasses import dataclass
from typing import Optional, Union, List,Tuple
from simple_parsing import ArgumentParser, subgroups, field
import simple_parsing

from dataset.dataset_arguements import DataloaderConfig,GroupDatasetConfig
from model.model_arguements import SignalModelConfig
from model.signal_model.Retnet.RetNetPool import SignalSeaConfig
from model.embedding.embedding_config import WaveletEmbeddingConfig
from train.train_arguements import TrainConfig, TaskConfig
from evaluator.evaluator_arguements import EvaluatorConfig, NormalInferenceMode, EvalPlotConfig,RecurrentInferenceMode, SlideInferenceMode
import json
from dataset.resource.resource_arguements import ResourceInstance
import dataclasses
 
from simple_parsing import NestedMode
from argparse import Namespace
from config.utils import print_namespace_tree
from simple_parsing.helpers.serialization import to_dict,save
from model import valid_model_config_pool
from trace_utils import print0
import accelerate

@dataclass
class ProjectConfig:
    
    DataLoader: DataloaderConfig
    task: TaskConfig = subgroups(
        {
            "train": TrainConfig,
            "infer": NormalInferenceMode,
            "recurrent_infer": RecurrentInferenceMode,
            "slide_infer":  SlideInferenceMode,
            "infer_plot": EvalPlotConfig
        }
    )
    model:  SignalModelConfig = subgroups(
        valid_model_config_pool|{'manual':SignalModelConfig},
        default='manual'
        )
    output_dir: str = field(default=None)
    trial_name: str = field(default='')
    accelerate_version: str = field(default='')
    custom_operation: bool = False ### debug or operation not train like generate mean-var information
    datapart:int=0
    def get_name(self):
        shape_string = "_".join( [str(self.history_length)] + [str(t) for t in self.img_size])
        return f'{self.model_type}.{shape_string}.{self.in_chans}.{self.out_chans}.{self.embed_dim}.{self.depth}.{self.num_heads}'

    def __post_init__(self):
        output_dir, dataset_name, model_name, Name =  get_the_correct_output_dir(self)
        self.output_dir   = output_dir
        self.dataset_name = dataset_name
        self.model_name   = model_name
        self.trial_name   = Name
        self.accelerate_version = accelerate.__version__

        if self.model.Predictor.use_confidence:
            if self.model.Predictor.use_confidence == 'whole_sequence':
                pass
            elif self.model.Predictor.use_confidence == 'status':
                assert self.DataLoader.Dataset.status_type in ['N0P1S2']
            elif self.model.Predictor.use_confidence == 'P-2to10':
                assert self.DataLoader.Dataset.status_type in ['P-2to10']
            elif self.model.Predictor.use_confidence == 'P0to5':
                assert self.DataLoader.Dataset.status_type in ['P0to5']
            elif self.model.Predictor.use_confidence == 'P-2to60':
                assert self.DataLoader.Dataset.status_type in ['P-2to60']
            elif self.model.Predictor.use_confidence == 'P-2to120':
                assert self.DataLoader.Dataset.status_type in ['P-2to120']
            else:
                assert self.model.Predictor.use_confidence == self.DataLoader.Dataset.status_type, f"""
                Predictor.use_confidence {self.model.Predictor.use_confidence} != Dataset.status_type {self.DataLoader.Dataset.status_type}. 
                Currently, use confidence means you will not using phase information for prediction and we will use the status_type as the confidence.
                """
        
        if self.model.Predictor.prediction_type=='slide_window':
            self.model.Backbone.max_length = self.DataLoader.Dataset.max_length
            # if isinstance(self.model.Embedding, WaveletEmbeddingConfig):
            #     self.model.Backbone.max_length = self.model.Embedding.embedding_length
        if isinstance(self.DataLoader.Dataset,GroupDatasetConfig):
            #print("========= here !!!!!!!!! ============")
            N = self.DataLoader.Dataset.max_station
            self.model.distance_system_dim = N*(N-1) + 1
        # if isinstance(self.DataLoader.Dataset.Resource.bandfilter_rate,str):
        #     if self.DataLoader.Dataset.Resource.bandfilter_rate == 'multiewn8':
        #         if self.model.Embedding.wave_channel != 24:
        #             print0(f"Warning: you are using multiewn8, but the wave_channel is not 24. We will set the wave_channel to 24")
        #         self.model.Embedding.wave_channel = 24
        #     else:
        #         raise NotImplementedError(f"the bandfilter_rate {self.DataLoader.Dataset.Resource.bandfilter_rate} is currectly implemented")
        # if 'findP' in self.model.Predictor.downstream_pool:
        #     assert self.model.Predictor.slide_feature_window_size is not None, "you need to specify the slide_feature_window_size for findP"
        #     ## then add slide_feature_window_size to the dataset thus the label is correct
        #     self.model.Embedding.use_phase = True
        if isinstance(self.model, SignalSeaConfig):
            assert self.DataLoader.Dataset.status_type == "N0P1S2" or 'status' in self.model.Predictor.downstream_pool, "you need to specify the status_type for SignalSeaConfig"

        if isinstance(self.model.Embedding, WaveletEmbeddingConfig):
            ### lets check the length and frequency
            assert self.model.Embedding.wave_length == self.DataLoader.Dataset.max_length, f"the wave_length {self.model.Embedding.wave_length} is not equal to the max_length {self.DataLoader.Dataset.max_length}"
            assert self.model.Embedding.wavelet_dt  == 1/self.DataLoader.Dataset.Resource.resource_frequence, f"the wavelet_dt {self.model.Embedding.wavelet_dt} is not equal to the sample_rate {1/self.DataLoader.Dataset.Resource.resource_frequence}"


import os,re
import time,hashlib
from datetime import datetime
def get_the_correct_output_dir(args: ProjectConfig):
    model_name   = args.model.name
    dataset_name = args.DataLoader.Dataset.name
    if isinstance(args.task, EvalPlotConfig):
        output_dir= args.task.plot_data_dir
        dataset_name_from_path = re.findall(r"checkpoints\/(.*?)\/", args.task.plot_data_dir)[0]
        if dataset_name_from_path != dataset_name:
            print0(f"Warning: the dataset_name {dataset_name} is not equal to the dataset_name in the plot_data_dir {dataset_name_from_path}. We will use the {dataset_name} assigned")
            #dataset_name = dataset_name_from_path
        path_in_list = output_dir.split('/')
        position_of_datasetname = path_in_list.index(dataset_name_from_path)
        model_name = path_in_list[position_of_datasetname+1]
        Name = path_in_list[position_of_datasetname+2]
        output_dir = '/'.join(path_in_list[:position_of_datasetname+3])
        
    elif (isinstance(args.task, TrainConfig) and args.task.continue_train) or isinstance(args.task, EvaluatorConfig):
        assert args.task.Checkpoint.preload_state or args.task.Checkpoint.preload_weight
        if isinstance(args.DataLoader.Dataset.Resource, ResourceInstance):
            WARNING_String= """after 2024_04_13 we rotate the origin angle named back_azimuth_deg as theta = theta - 180. thus, it should be forbidden to use the pytorch model or checkpoint before that
             unless we mannual modifty the weight(pytorch_model.bin).
             
             For most case, it will require you add minus sign (+) --> (-) for the last layer weight of x and y branch in pytorch_model.bin, 
             """
            weight_path = (args.task.Checkpoint.preload_state or args.task.Checkpoint.preload_weight)
            check_date = datetime(2024, 4, 13)
            do_check = datetime.fromtimestamp(os.path.getmtime(weight_path)) < check_date
            if not do_check: 
                do_check   = datetime.fromtimestamp(os.path.getmtime(os.path.dirname(weight_path))) < check_date
            if not do_check:
                trial_name = args.task.Checkpoint.preload_state.rstrip('/').split('/')[-3] if args.task.Checkpoint.preload_state else args.task.Checkpoint.preload_weight.split('/')[-2]
                if '_' in trial_name and '-' in trial_name:
                    datelist = trial_name.split('-')[0].split('_')
                    if len(datelist) > 2:
                        month, day = datelist[:2]
                        year = 2023 if month>4 else 2024
                        do_check   = datetime(year, int(month), int(day)) < check_date
            if do_check:
                assert args.task.Checkpoint.preload_state is None or len(args.task.Checkpoint.preload_state), """
                    Since the weight should be modifed, the preload state is unavilable
                    """
                the_converted_mark = weight_path.rstrip('/')+'.checked'
                assert os.path.exists(the_converted_mark), WARNING_String
            else:
                print("A Simple Notice for the usage about [Instance], the back_azimuth_deg in metadata before 2024-04-13 is actually the azimuth who use the event postition coordinate. Be careful for using the weight before 2024-04-13")
        if args.task.Checkpoint.preload_state:
            output_dir = os.path.dirname(os.path.dirname(args.task.Checkpoint.preload_state))
        elif args.task.Checkpoint.preload_weight:
            output_dir = os.path.dirname(args.task.Checkpoint.preload_weight)
        
        Name = [part for part in output_dir.split("/") if '-seed_' in part ]
        if len(Name)==1:
            Name= Name[0]
        else:
            Name = output_dir.strip("/").split("/")[-1]
        if isinstance(args.task, EvaluatorConfig):
            dataset_name_from_path = re.findall(r"checkpoints\/(.*?)\/", output_dir)[0]
            if dataset_name_from_path != dataset_name:
                print0(f"Warning: the dataset_name {dataset_name} is not equal to the dataset_name in the plot_data_dir {dataset_name_from_path}. We will use the dataset_name assigned")
    else:
        if args.task.Checkpoint.preload_state and not args.task.continue_train:
            print0("we don't continue train. We just start from a pretrained weight ")
        if args.trial_name:
            Name = args.trial_name
        else:
            # Name = time.strftime("%m_%d_%H_%M_%S") ###<-- there is a bug here. when multinode training
            rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
            local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
            rank = rank + local_rank
            confighash = hashlib.md5(str(args).encode("utf-8")).hexdigest()
            cachefile = "log/records"
            recordfile = os.path.join(cachefile, confighash)
            if rank == 0:
                if not os.path.exists(cachefile):os.makedirs(cachefile)
                Name = time.strftime("%m_%d_%H_%M")
                Name = f"{Name}-seed_{args.task.seed}-{confighash[:4]}"
                with open(recordfile, "w") as f:
                    json.dump({'Name':Name}, f)
            else:
                time.sleep(1)
                time_start = time.time()
                while not os.path.exists(recordfile):
                    time_cost =  time.time() - time_start
                    if time_cost > 60: raise TimeoutError(f"rank {rank} can't find the recordfile {recordfile} in 10s")
                with open(recordfile, "r") as f:
                    Name = json.load(f)['Name']
            #print(f"GPU:{rank}:Name:{Name}")
        output_dir = f'checkpoints/{dataset_name}/{model_name}/{Name}'
        
    dataset_name = dataset_name.split('_')[0] # remove all the subfix like _sample6000 
    return output_dir, dataset_name, model_name, Name


def get_parallel_config_of_accelerator(accelerator):
    parallel_config = {}
    for key, val in accelerator.state._shared_state.items():
        if isinstance(val, (list, tuple, str, int, bool)) or val is None:
            parallel_config[key] = val
        elif key in ['deepspeed_plugin']:
            parallel_config[key] = to_dict(val)
            parallel_config[key]['hf_ds_config'] = parallel_config[key]['hf_ds_config'].config
        else:
            accelerator.print(f"skip unserializable key {key}={val}")
    return parallel_config
# def get_args(config_path=None):
#     args = simple_parsing.parse(config_class=ProjectConfig, config_path=None, args=None, add_config_path_arg=True)
#     return args
def dict_to_arglist(d):
    arglist = []
    for k, v in d.items():
        arglist.append('--' + str(k))
        if v is not None:
            if isinstance(v, (list,tuple)):
                for vvv in v:
                    arglist.append(str(vvv))
            else:
                arglist.append(str(v))


        else:
            arglist.pop(-1)
    return arglist

def old_version_config_clean(defaults):
    if 'trial_name' in defaults: 
        defaults.pop('trial_name')
    if 'infer' in defaults:
        defaults.pop('infer')
    if 'infer_plot' in defaults:
        defaults.pop('infer_plot')


def get_args(config_path=None, args=None)->ProjectConfig:

    conf_parser = ArgumentParser(add_help=False)
    conf_parser.add_argument("-c", "--conf_file",  default=None, help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args(args)
    config_path = config_path if config_path else args.conf_file
    defaults = {}
    if config_path:
        with open(config_path, 'r') as f:defaults = json.load(f)
        defaults['use_wandb'] = False
    the_old_version_config = config_path is not None and 'DataLoader' not in defaults
    if the_old_version_config:
        print0("WARNING: you are using the old version config file. This will be deprecated in the future.")
        if 'model' not in defaults: 
            if 'model_config_name' in defaults:
                defaults['model'] = modelalias(defaults.pop('model_config_name'))
            else:
                defaults['model'] = modelalias(defaults.pop('model_type'))
            
        if 'resource_source' not in defaults: 
            defaults['resource_source']=defaults.pop('dataset_name')
        old_version_config_clean(defaults)
        
        form_args =  dict_to_arglist(defaults)
        whole_args= form_args + remaining_argv # put the remaining argv at the end, thus that can overwrite the default config
        
        parser = ArgumentParser()
        parser.add_arguments(ProjectConfig, dest="config")
        args,remaining_argv = parser.parse_known_args(whole_args)
        if len(remaining_argv) > 0:
            print0("Warning: some arguements are not parsed: ", remaining_argv)
            if 'train' in args:
                assert '--freeze_mode' not in remaining_argv, "--freeze_mode is deprecated. Please use task.Freeze(freeze_embedder=?,freeze_backbone=?,freeze_downstream=?) instead"
        return args.config
    else:
        assert config_path is None, "the new version config file is not supported yet"
        parser = ArgumentParser(nested_mode=NestedMode.WITHOUT_ROOT,config_path=config_path)
        parser.add_arguments(ProjectConfig, dest="config")
        args = parser.parse_args(remaining_argv)
        return args.config

def modelalias(name):
    if name  =='GoldfishS40MA':
        return 'GoldfishS40M'
    return name