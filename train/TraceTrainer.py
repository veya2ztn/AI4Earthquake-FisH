
from __future__ import annotations

from typing import Any, Callable, Sequence, Sized

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
#from .timer import Timers

import time,os
import numpy as np
from transformers.trainer_pt_utils import distributed_broadcast_scalars
from mltool.dataaccelerate import DataSimfetcher
import json
from trace_utils import get_local_rank
from accelerate.utils import DistributedType
from .utils import DummyProgressBar, DistributedTqdmProgressBar, LossTracker
from .train_arguements import TrainConfig
from accelerate.data_loader import DataLoaderShard, DataLoaderDispatcher
import accelerate

from evaluator.trace_evaluator import dict_to_dict_of_lists, dict_of_lists_to_numpy, findAllPS, metrix_PS
import scipy
from trace_utils import print0
from evaluator.utils import findAllP_Peak, picking
from project_arguements import get_parallel_config_of_accelerator
import traceback

class MeanVarComputing:
    def __init__(self, shape):
        self.shape = shape
        self.count = 0
        self.mean  = None
        self.M2    = None

    def update(self, new_data_batch):
        # new_data_batch should have shape (batch_size, W, H, C)
        if new_data_batch.shape[1:] != self.shape:
            raise ValueError(f"Data shape must be {self.shape}, but got {new_data_batch.shape[1:]}")

        batch_size = new_data_batch.shape[0]
        data_mean  = torch.mean(new_data_batch, dim=0)
        data_var   = torch.var(new_data_batch , dim=0) * (batch_size - 1)
        
        if self.count == 0:  # If this is the first batch, initialize mean and M2 directly
            self.mean = data_mean
            self.M2   = data_var
        else:  # Update existing mean and M2
            total_count = self.count + batch_size
            delta = data_mean - self.mean
            self.mean += delta * batch_size / total_count
            self.M2 += data_var + delta**2 * self.count * batch_size / total_count

        self.count += batch_size

    def finalize(self):
        if self.count < 2:
            raise ValueError("Variance calculation requires at least two data points")
        self.variance = self.M2 / (self.count - 1)
        return self.mean, self.variance

    def get_mean(self):
        return self.mean

    def get_variance(self):
        if self.count < 2:
            raise ValueError("Variance calculation requires at least two data points")
        return self.M2 / (self.count - 1)


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader | None ,
        optimizer: Optimizer,
        lr_scheduler,
        accelerator: Accelerator,
        train_config: TrainConfig,
        epoch_end_callbacks: Sequence[Callable[['Trainer'], None]] | None = None,
    ):
        self.train_config = train_config
        self.model = model
        
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.not_valid_during_train = train_config.not_valid_during_train
        self.epochs = train_config.epochs
        self.log_interval = train_config.Monitor.log_interval
        self.save_on_epoch_end = train_config.save_on_epoch_end
        self.clip_value = train_config.clip_value
        
        self.train_loss_tracker      = LossTracker()
        self.validation_loss_tracker = LossTracker()
        # if isinstance(self.train_dataloader.dataset, Sized):
        #     num_steps_per_epoch = len(self.train_dataloader)
        # else:
        #     num_steps_per_epoch = None
        num_steps_per_epoch = len(self.train_dataloader)
        self.progress_bar = DistributedTqdmProgressBar(self.epochs, num_steps_per_epoch=num_steps_per_epoch,
                                                       bar_log_path=os.path.join(self.accelerator.project_dir,'log',f'train_bar.log'), 
                                                       desc="Training....",)
        self.valid_progress_bar = DistributedTqdmProgressBar(self.epochs, num_steps_per_epoch=len(self.validation_dataloader), 
                                                    desc="Validating....",)
        self.epoch_end_callbacks = epoch_end_callbacks or []
        self.current_step = 0
        #self.timers = Timers()
        self.best_validation_loss = np.inf
        self.best_validation_metrics = {}
        weight_dir   = os.path.join(self.accelerator.project_dir, 'best', 'weight')
        os.makedirs(weight_dir, exist_ok=True)
        self.cached_weight_dir = weight_dir
        self.metric_report_path = os.path.join(self.accelerator.project_dir,'metric_report.json')
           
    def set_train_mode(self, mode='train'):
        
        self.model.train()
        model = self.accelerator.unwrap_model(self.model)
        model.freeze_model_during_train(self.train_config.Freeze)

    def compute_var_mean(self, root, shape=(3,6000)): #root = "datasets/DiTing330km/diting.group.good.256.2048.wavelet"
        self.progress_bar.on_epoch_start(0)
        mean_var_computer = MeanVarComputing((3, 6000))
        model = self.accelerator.unwrap_model(self.model)
        index = self.accelerator.process_index
        with torch.no_grad():
            for batch_index, batch in enumerate(self.train_dataloader):
                waveform = model.wave_embedding[0](batch['waveform_seq'])
                self.progress_bar.update(1)
                mean_var_computer.update(waveform)
            
        self.progress_bar.on_epoch_end()    
        mean, variance = mean_var_computer.finalize()
        mean = mean.detach().cpu().numpy()
        variance= variance.detach().cpu().numpy()
        
        
        os.makedirs(root,  exist_ok=True)
        np.save(f'{root}/mean.{index}.npy', mean)
        np.save(f'{root}/std.{index}.npy', variance)
        self.accelerator.wait_for_everyone()

    def train(self, start_epoch=0):
        validation_loss=None
        update_intervel = max(len(self.train_dataloader)//100,2)
        nan_count = 0
        
        for current_epoch in range(0, self.epochs + 1):
            if current_epoch < start_epoch:continue
            if self.lr_scheduler is not None and current_epoch>1:self.lr_scheduler.step(current_epoch-1)
            if not self.train_config.do_validation_at_first_epoch and current_epoch==0:continue
            #self.model.train()
            if current_epoch >= 1:
                self.set_train_mode()
                self.progress_bar.on_epoch_start(current_epoch)
                data_loading = []
                model_train = []
                #self.timers('data_loading').start()
                featcher = DataSimfetcher(self.train_dataloader,device=self.accelerator.device)
                last_record_time = time.time()
                failed_times=  0
                
                # for batch_index in range(len(self.train_dataloader)):
                #     batch = featcher.next()
                if not isinstance(self.train_dataloader, (DataLoaderShard,DataLoaderDispatcher)):
                    self.train_dataloader.sampler.set_epoch(current_epoch)
                for batch_index, batch in enumerate(self.train_dataloader):
                    # if batch_index == 0: print(f"{self.accelerator.process_index}=>{batch['idx']}")
                    # continue
                    batch = self.auto_precision(batch)
                    data_loading.append(time.time() - last_record_time);last_record_time =time.time() 
                    #<-- In accelerate mode, the time record will be quite large and the model time will be quite small
                    #<-- However, the totally cost remain, thus it is a typical bug inside accelerate.
                    
                    with self.accelerator.accumulate(self.model):
                        self.optimizer.zero_grad()
                        batch_output = self.model(**batch)
                        loss = batch_output['loss']
                        
                        if not torch.isnan(loss):
                            nan_count = 0
                        else:
                            nan_count+=1 
                            assert nan_count < 10, f"too many nan detected, exit"
                        self.train_loss_tracker.update(loss) 
                        #<---- the accelerator.backward will change the loss itself for RWKV model, 
                        #<---- it not effect on training ? Really ?
                        #<---- RWKV model of huggingface not support the bf16 mode as the timedecay parameter is not bf16 when using bf16
                        #print(f"{loss.item()}", end=" ")
                        self.accelerator.backward(loss)
                        #print(f"{loss.item()}",end=" ")
                        #self.train_loss_tracker.update(loss)
                        #print(f"{loss.item()} == >{self.train_loss_tracker.loss}")
                        if self.accelerator.sync_gradients and self.clip_value:
                            self.accelerator.clip_grad_value_(self.model.parameters(), self.clip_value)
                        self.optimizer.step()

                        
                    model_train.append(time.time() - last_record_time);last_record_time =time.time()
                    
                    if batch_index%update_intervel==1:self.progress_bar.update(update_intervel)
                    self.current_step += 1
                    
                    if batch_index % self.log_interval == 0 or batch_index < 30:
                        log_dict = {'loss': self.train_loss_tracker.loss,
                                    'data': np.mean(data_loading),
                                    'model': np.mean(model_train),
                                    #'time':self.timers.get_string()
                                    }
                        for key, val in batch_output['error_record'].items():log_dict[key] = val
                        show_dict = dict([[k,v] for k,v in log_dict.items() if "e_" not in k])                    
                        self.progress_bar.show_metrics(show_dict)
                        train_metrics = self.add_prefix(log_dict, 'itering')
                        self.accelerator.log(train_metrics, step=self.current_step)
                        data_loading = []
                        model_train = []

                    last_record_time = time.time()
                        
                    #self.timers('data_loading').start()
                    if self.train_config.time_test and batch_index>50:break
                self.accelerator.wait_for_everyone()

                if self.train_config.time_test:
                    if self.accelerator.is_main_process:
                        current_rate = self.progress_bar.progress_bar.format_dict['rate']
                        current_speed= current_rate*self.batch_size
                        if os.path.exists('speed_test.json'):
                            with open('speed_test.json', 'r') as f: 
                                speed_record = json.load(f)
                        else:
                            speed_record = {}
                        current_processing = self.accelerator.num_processes
                        parallel_config    = get_parallel_config_of_accelerator(self.accelerator)
                        distributed_type = parallel_config['distributed_type']
                        model_type = self.model_name
                        if model_type not in speed_record:
                            speed_record[model_type]={}
                        if distributed_type not in speed_record[model_type]:
                            speed_record[model_type][distributed_type]={}
                        if current_processing not in speed_record[model_type][distributed_type]:
                            speed_record[model_type][distributed_type][current_processing]={}
                        if self.batch_size not in speed_record[model_type][distributed_type][current_processing]:
                            speed_record[model_type][distributed_type][current_processing][self.batch_size]=[]
                        data_type = parallel_config['_mixed_precision'] if distributed_type is not DistributedType.DEEPSPEED else "bf16"
                        speed_record[model_type][distributed_type][current_processing][self.batch_size].append([data_type, current_speed])
                        setting = f"SET:{model_type}|{distributed_type}.{data_type}.G{current_processing}.B{self.batch_size}"
                        print(f"{setting} => {current_speed}")
                        os.system(f"""echo "{setting} => {current_speed}" >> speed_test.record.log """)
                        with open('speed_test.json', 'w') as f: 
                                json.dump(speed_record, f)
                    raise

                train_metrics = self.add_prefix({'loss': self.train_loss_tracker.loss, 
                                                'lr':self.optimizer.param_groups[0]['lr']}, 'train')
                
                train_metrics['epoch'] = current_epoch
                self.accelerator.log(train_metrics, step=self.current_step)
                #self.accelerator.log(1)
                self.train_loss_tracker.on_epoch_end()
                self.progress_bar.on_epoch_end()
                try:
                    self.save_state(current_epoch)
                except:
                    traceback.print_exc()
                    #print("Failed to save the state")
                    pass
            
            

            if self.validation_dataloader and not self.not_valid_during_train:
                validation_loss=None
                error_pool = {}
                ######### collect validation infomration #########
                error_pool = self.evaluate(
                    self.validation_dataloader,
                    self.validation_loss_tracker
                )
                error_pool = self._nested_gather_scalar_dict(error_pool)
                validation_loss  = error_pool['loss'] 
                validation_metrics_pool = error_pool
                validation_metrics = self.add_prefix(validation_metrics_pool, 'validation')
                validation_metric_string = "\n".join([f'    {key}: {val:.4f}' for key, val in validation_metrics.items()])
                self.accelerator.print(f'Epoch {current_epoch}:\n{validation_metric_string}')
                
                
                validation_metrics['epoch'] = current_epoch
                self.accelerator.log(validation_metrics, step=self.current_step)
                ######### save the best weight #########
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.update_best_state_information(validation_metrics,current_epoch)
                    self.maintain_the_cached_weight(current_epoch)
                
            
            
            handle = self.train_dataloader.dataset.waveform_handle ### close the handle since all data loaded into memeory
            if hasattr(handle,'reset'): handle.reset()
            if self.epoch_end_callbacks:
                for callback in self.epoch_end_callbacks:
                    callback(self)
            self.accelerator.wait_for_everyone()
            #torch.cuda.empty_cache()
            
        #self.accelerator.log(self.add_prefix(self.best_validation_metrics, 'best'))
        
        
        if self.train_config.clean_checkpoints_at_end and self.accelerator.is_main_process:
            the_checkpoints_path = os.path.join(self.accelerator.project_dir, 'checkpoints', f'checkpoints_{self.epochs-1}')
            os.system(f'rm -rf {the_checkpoints_path}/*')
            os.system(f'touch {the_checkpoints_path}/cleanup_due_to_finish_train')

    def save_state(self,epoch):
        if epoch % self.train_config.Checkpoint.save_every_epoch != 0:return
        if not self.save_on_epoch_end:return 
        #self.accelerator.project_configuration.iteration = epoch #<--- semem will save checkpoint every epoch
        # Should enable all process for DEEPSPEED: https://github.com/huggingface/diffusers/issues/2606
        self.accelerator.save_state(safe_serialization=False) #<<---- please do not use the safe_serialization=True
        # if self.accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM] or int(accelerate.__version__.split('.')[-2])<24:
        #     self.accelerator.save_state()
        # else:
        #    if self.accelerator.is_main_process:
        #         self.accelerator.save_state()
        # return
    
    def _nested_gather_scalar_dict(self, _dict):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        new_dict = _dict
        if _dict is None:
            return 
        #### try to sync the distributed tensor and compute the correct error cross GPU. Fail! ####
        if (self.accelerator.distributed_type != DistributedType.NO):
            new_dict = {}
            for key, scalar in _dict.items():
                new_dict[key]=torch.mean(distributed_broadcast_scalars([scalar])).item()
        return new_dict

    def get_metric_report(self):
        metric_report_path = self.metric_report_path 
        if os.path.exists(metric_report_path):
            with open(metric_report_path, 'r') as f: 
                now_metric_report = json.load(f)
        else:
            now_metric_report = {}
        return now_metric_report
    
    def update_metric_report(self, key, val, epoch, weight_path):
        metric_report_path = self.metric_report_path 
        now_metric_report = self.get_metric_report()
        now_metric_report[key]   = {
            'score':val,
            'epoch':epoch,
            'path':weight_path
        }
        with open(metric_report_path, 'w') as f:
            json.dump(now_metric_report, f, indent=4)

    def auto_precision(self, torch_pool):
        if self.accelerator.state.mixed_precision !='bf16':return torch_pool
        if self.accelerator.distributed_type not in [DistributedType.FSDP, DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM]:return torch_pool
        
        dtype = torch.bfloat16 if self.accelerator.state.mixed_precision == "bf16" else torch.float32
        for key,val in torch_pool.items():
            if isinstance(val,torch.Tensor):
                torch_pool[key] = val.to(dtype=dtype)
            elif isinstance(val,dict):
                torch_pool[key] = self.auto_precision(val)
            elif isinstance(val,list):
                torch_pool[key] = [self.auto_precision(v) for v in val]
            elif isinstance(val,tuple):
                torch_pool[key] = tuple([self.auto_precision(v) for v in val])
        return torch_pool
        

    def update_best_state_information(self, validation_metrics: dict, current_epoch: int):
        for key in validation_metrics.keys():
            if key in ['epoch']:continue
            if "loss_" in key: continue
            if 'realtime' in key:continue
            if 'posttime' in key:continue # do not save the metric like validation.posttime.s.ws3.t0.recall_beta.at0.5
            if 'record' in key: continue
            self.update_best_state_by_key(
                key, validation_metrics, current_epoch)
    
    def update_best_state_by_key(self, key: str, validation_metrics: dict, current_epoch: int):
        large_is_better = (key.split('/')[-1][0]=='a') or ('precision' in key) or ('recall' in key)
        small_is_better = not large_is_better
        
        if key not in validation_metrics:
            raise ValueError(f'Key {key} not found in validation metrics')
        validation_loss      = validation_metrics[key]
        if key not in self.best_validation_metrics:
            self.best_validation_metrics[key]=np.inf if small_is_better else 0
        best_validation_loss = self.best_validation_metrics[key]
        goodQ = (validation_loss < best_validation_loss) if small_is_better else (validation_loss > best_validation_loss)
        if (validation_loss and  goodQ):
            save_dir = os.path.join(self.accelerator.project_dir, 'best', key.strip('/').replace('/','.'))
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, 'score'), 'a') as f:f.writelines(f"{key} {current_epoch} {validation_loss}\n")
            linked_checkpoint_path = self.get_current_cached_weight_path(current_epoch)
            self.update_metric_report(key, self.best_validation_metrics[key], 
                                      current_epoch, linked_checkpoint_path)
            self.create_soft_link(linked_checkpoint_path, os.path.join(save_dir, 'checkpoint'))
        if small_is_better:
            self.best_validation_metrics[key] = min(validation_loss, best_validation_loss)
        else:
            self.best_validation_metrics[key] = max(validation_loss, best_validation_loss)

    def create_soft_link(self, linked_checkpoint_path, softlink_of_checkpoints):
        #linked_checkpoint_path = "checkpoints/stead.trace.BDLEELSSO_1000of3000/ViT_MS_A.3.512.1024.256.4/10_11_17_21-seed_21/best/weight/epoch-7.bin"
        #softlink_of_checkpoints = "checkpoints/stead.trace.BDLEELSSO_1000of3000/ViT_MS_A.3.512.1024.256.4/10_11_17_21-seed_21/best/validation.loss_x/checkpoint.bin"
        relative_path = os.path.relpath(linked_checkpoint_path,os.path.dirname(softlink_of_checkpoints))
        os.system(f"rm {softlink_of_checkpoints}; ln -s {relative_path} {softlink_of_checkpoints}")
        
        #print(softlink_of_checkpoints)
        #os.symlink(relative_path,softlink_of_checkpoints)
    
    def get_current_cached_weight_path(self, current_epoch):
        return os.path.join(self.cached_weight_dir,f'epoch{current_epoch:04d}')

    def get_used_checkpoint_path(self):
        now_metric_report = self.get_metric_report()
        return [pool['path'] for pool in now_metric_report.values()]
    
    def maintain_the_cached_weight(self, current_epoch):
        if current_epoch < 1: return
        existed_checkpoint_path = [os.path.join(self.cached_weight_dir, p) for p in os.listdir(self.cached_weight_dir)]
        used_checkpoint_path    = self.get_used_checkpoint_path()
        now_checkpoint_path     = self.get_current_cached_weight_path(current_epoch)
        if now_checkpoint_path in used_checkpoint_path:
            if now_checkpoint_path not in existed_checkpoint_path:
                unwrapper_model = self.model
                while hasattr(unwrapper_model,'module'):
                    unwrapper_model = unwrapper_model.module
                unwrapper_model.save_pretrained(now_checkpoint_path,safe_serialization=False)
        for path in existed_checkpoint_path:
            if path not in used_checkpoint_path:
                ## clean the unused checkpoint
                print(f"clean the unused checkpoint {path}")
                os.system(f'rm -r {path}')

        
        

    def log_metrics(self, metrics, step: int, flag='iter'):

        self.accelerator.log(metrics, step=step)
        self.progress_bar.show_metrics(metrics)

    @staticmethod
    def add_prefix(values: dict[str, Any], prefix: str):
        return {f'{prefix}/{k}': v for k, v in values.items()}


    def evaluate(
        self,
        dataloader: DataLoader,
        loss_tracker: LossTracker | None = None
    ):
        update_intervel = 10
        self.model.eval()
        loss_tracker = loss_tracker if loss_tracker is not None else LossTracker()
        error_tracker= {}
        failed_times = 0
        featcher = DataSimfetcher(dataloader, device=self.accelerator.device)
        # for batch_index in range(len(dataloader)):
        #     batch = featcher.next()
        tracker = {
            'status':{},
            'probabilityPeak':{},
            'findP':{},
            'findS':{},
            'P_Peak_prob':{}
        }
        tracker_keys = list(tracker.keys())
        self.valid_progress_bar.on_epoch_start()
        for batch_index,batch in enumerate(dataloader):
            if batch is None:continue
            with torch.inference_mode():
                batch = self.auto_precision(batch)
                batch_output = self.model(get_prediction='pred+real', **batch)
                prediction   = batch_output['prediction']
                error_record = batch_output['error_record']
                loss_tracker.update(batch_output['loss'])
                for key, val in error_record.items():
                    if key not in error_tracker:error_tracker[key] = LossTracker()
                    error_tracker[key].update(val)
            
            for key in tracker_keys:
                if key not in prediction:continue
                dict_to_dict_of_lists(tracker[key], {key:prediction[key]})

            if batch_index%update_intervel==1:self.valid_progress_bar.update(update_intervel)
            #if self.train_config.do_validation_at_first_epoch and batch_index > 10:break
        self.valid_progress_bar.on_epoch_end()
        loss = loss_tracker.loss
        loss_tracker.on_epoch_end()
        out = dict([[k,t.loss] for k,t in error_tracker.items()])
        out['loss'] = loss
        key = 'status'
        if len(tracker[key])>0:
            tracker[key] = dict_of_lists_to_numpy(tracker[key], dim=0, mode='concat')
            self.get_accurancy_for_status(tracker[key][key], dataloader, out)   
        
        key = 'probabilityPeak'
        if len(tracker[key])>0:
            tracker[key] = dict_of_lists_to_numpy(tracker[key], dim=0, mode='concat')
            self.get_accurancy_for_probabilityPeak(tracker[key][key], dataloader, out)


        key = 'findP'
        if len(tracker[key])>0:
            tracker[key] = dict_of_lists_to_numpy(tracker[key], dim=0, mode='concat')
            self.get_accurancy_for_findP(tracker[key][key], dataloader, out)

        key = 'findS'
        if len(tracker[key])>0:
            tracker[key] = dict_of_lists_to_numpy(tracker[key], dim=0, mode='concat')
            self.get_accurancy_for_findP(tracker[key][key], dataloader, out)

        key = 'P_Peak_prob'
        if len(tracker[key])>0:
            tracker[key] = dict_of_lists_to_numpy(tracker[key], dim=0, mode='concat')
            self.get_accurancy_for_P_Peak_prob(tracker[key][key], dataloader, out)
        return out
    


    def get_accurancy_for_findP(self,  findP, dataloader, out):
        model             = self.accelerator.unwrap_model(self.model)
        real = findP['real']
        pred = findP['pred']
        if len(findP['pred'].shape)==3:
            B, N, L = findP['pred'].shape
            real = real.reshape(B*N)
            pred = pred.reshape(B*N, L)
        Plocation         = np.argmax(pred, axis=1) - 1 ## (B,) -1 means no P and other value means the P location in the window
        real_position     = real# (B, ) float
        sequence_length   = pred.shape[1] - 1 # L+1 to L
        real_slot         = np.round(sequence_length*real_position)
        p_arrival_samples = real_slot*model.embedder_config.resolution
        Plocation         = Plocation*model.embedder_config.resolution
        ppicks={}
        for row, pred in enumerate(Plocation):
            ppicks[row] = set([pred.item()])
        
        counting_type= 'findP'
        dataset = dataloader.dataset
        metric_curve_p = metrix_PS(pred=ppicks, target=p_arrival_samples,
                        freq= dataset.config.Resource.sampling_frequence,
                        max_length= dataset.max_length ,
                        flag=f'{counting_type}',verbose=False,
                        metric_types = ['alpha'])
        for key, val in metric_curve_p.items():
            out[key] = val

    def get_accurancy_for_status(self, status,dataloader, out  ):
        max_filter_time = 0.5
        dataset = dataloader.dataset
        model = self.accelerator.unwrap_model(self.model)
        frequency =  dataset.config.Resource.sampling_frequence
        resolution = model.embedder_config.resolution
        max_filter_window_size = (max_filter_time*frequency)//resolution
        if max_filter_window_size%2==0:max_filter_window_size+=1
        filter_time = max_filter_window_size*resolution/frequency
        probability_threshold  = 0
        pred_status = status['pred'] # (B, L//3, 3)
        real_status = status['real'] # (B, L//3, )
        pred_status = np.argmax(pred_status, axis=-1) # (B, L//3)

        model = self.accelerator.unwrap_model(self.model)


        ppicks_pool, spicks_pool = findAllPS(pred_status=pred_status,  ps_win=max_filter_window_size, expansion= resolution)
        preals_pool, sreals_pool = findAllPS(pred_status=real_status,  ps_win=1, expansion= resolution)
        
        
        
        
        p_arrival_samples = []
        s_arrival_samples = []
        for row in range(real_status.shape[1]):
            if row in preals_pool['realtime']:
                ppool = preals_pool['realtime'][row]
                assert len(ppool) == 1, f"len(pool) must be 1 (label must only contain one p-peak and one s-peak), but it is {len(ppool)}."
                p_arrival_samples.append(list(ppool)[0])
            else:
                p_arrival_samples.append(-1)
            if row in sreals_pool['realtime']:
                spool = sreals_pool['realtime'][row]
                assert len(spool) == 1
                s_arrival_samples.append(list(spool)[0])
            else:
                s_arrival_samples.append(-1)
        p_arrival_samples = np.array(p_arrival_samples)
        s_arrival_samples = np.array(s_arrival_samples)

        for counting_type in ppicks_pool.keys():
            ppicks = ppicks_pool[counting_type]
            spicks = spicks_pool[counting_type]
            dataset = dataloader.dataset
            
            metric_curve_p = metrix_PS(pred=ppicks, target=p_arrival_samples,
                        freq= frequency,
                        max_length= dataset.max_length ,
                        flag=f'{counting_type}.p/wt{filter_time}.t{probability_threshold}',verbose=False )
            metric_curve_s = metrix_PS(pred=spicks, target=s_arrival_samples,
                    freq= frequency,
                    max_length= dataset.max_length ,
                    flag=f'{counting_type}.s/wt{filter_time}.t{probability_threshold}',verbose=False )
            for key, val in metric_curve_p.items():
                out[key] = val
            for key, val in metric_curve_s.items():
                out[key] = val
    
    def get_accurancy_for_P_Peak_prob(self, P_Peak_prob, dataloader, out):
        max_filter_time = 0.5
        model = self.accelerator.unwrap_model(self.model)
        dataset = dataloader.dataset
        frequency =  dataset.config.Resource.sampling_frequence
        resolution = model.embedder_config.resolution
        max_filter_window_size = max_filter_time*frequency//resolution
        if max_filter_window_size%2==0:max_filter_window_size+=1
        filter_time = max_filter_window_size*resolution/frequency

        model                   = self.accelerator.unwrap_model(self.model)
        pred_status_probability = P_Peak_prob['pred'] # (B, L//3) 
        #print(pred_status_probability.shape)
        real_status_probability = P_Peak_prob['real'] # (B, L//3)
        assert pred_status_probability.shape == real_status_probability.shape
        #pred_status_probability = scipy.special.expit(pred_status_probability) ### output pred_status_probability is get sigmoid
        real_status             = (real_status_probability > 0.9).astype('int') # (B, L//3) # [0000000010000000000000]
        p_arrival_samples = []
        for row in real_status:
            ppos = np.where(row==1)[0]
            if len(ppos)>0:
                p_arrival_samples.append(ppos[0]*resolution)
            else:
                p_arrival_samples.append(-1)
            
        p_arrival_samples = np.array(p_arrival_samples)

        counting_type = 'p-peak'

        tri_th_h = 25/100
        tri_th_l = 15/100
        ppick = {}
        for row, data in enumerate(pred_status_probability):
            p_picking = picking(data,tri_th_h, tri_th_l )
            if len(p_picking)>0:
                p_picking = [p_picking[np.argmax(data[p_picking])]]
                p_picking = [t*resolution for t in p_picking]
                ppick[row] = set(p_picking)
        ppicks  = ppick
        dataset = dataloader.dataset
        
        metric_curve_p = metrix_PS(pred=ppicks, target=p_arrival_samples,
                    freq= frequency,
                    max_length= dataset.max_length ,
                    flag=f'{counting_type}.p',verbose=False )
        for key, val in metric_curve_p.items():
            out[key] = val
        
    def get_accurancy_for_probabilityPeak(self,probabilityPeak, dataloader, out):
        max_filter_time = 0.5
        model = self.accelerator.unwrap_model(self.model)
        dataset = dataloader.dataset
        frequency =  dataset.config.Resource.sampling_frequence
        resolution = model.embedder_config.resolution
        max_filter_window_size = max_filter_time*frequency//resolution
        if max_filter_window_size%2==0:max_filter_window_size+=1
        filter_time = max_filter_window_size*resolution/frequency

        model                   = self.accelerator.unwrap_model(self.model)
        
        real_status_probability = probabilityPeak['real'] # (B, L//3, 3)
        # find the correct position of P and S
        p_arrival_samples = []
        s_arrival_samples = []
        for row in real_status_probability:
            ppos = np.argmax(row[...,1])
            if row[ppos,1]>0.7:
                p_arrival_samples.append(ppos*resolution)
            else:
                p_arrival_samples.append(-1)
            spos = np.argmax(row[...,2])
            if row[spos,2]>0.7:
                s_arrival_samples.append(spos*resolution)
            else:
                s_arrival_samples.append(-1)
        p_arrival_samples = np.array(p_arrival_samples)
        s_arrival_samples = np.array(s_arrival_samples)
        
        pred_status_probability = probabilityPeak['pred'] # (B, L//3, 3)
        pred_status_probability = scipy.special.softmax(pred_status_probability,-1)
        counting_type = 'peak'
        #pred_status = np.argmax(pred_status, axis=-1) # (B, L//3) # [0000000010000002000000] 
        #for probability_threshold in [0.3, 0.5, 0.7, 0.9]:
        ppicks, spicks= findAllP_Peak(pred_status_probability,0.35,0.15,expansion=resolution)
        dataset = dataloader.dataset
        
        metric_curve_p = metrix_PS(pred=ppicks, target=p_arrival_samples,
                    freq= frequency,
                    max_length= dataset.max_length ,
                    flag=f'{counting_type}.p',verbose=False )
        metric_curve_s = metrix_PS(pred=spicks, target=s_arrival_samples,
                freq= frequency,
                max_length= dataset.max_length ,
                flag=f'{counting_type}.s',verbose=False )
        for key, val in metric_curve_p.items():
            out[key] = val
        for key, val in metric_curve_s.items():
            out[key] = val

