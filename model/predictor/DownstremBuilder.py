import os
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from einops.layers.torch import Rearrange
from .criterion import *
from .predictor_config import PredictorConfig
from ..signal_model.backbone_config import BackboneConfig
from ..signal_model.Retnet.retnet.self_retention import RMSNorm
from model.utils import  compute_accu_matrix
from dataset.normlizer import get_normlizer_convert, _NormlizerConvert
from trace_utils import print0
NonLinearPool = {
    'silu':nn.SiLU,
    'sigmoid':nn.Sigmoid,
    'none':nn.Identity,
    'selu':nn.SELU,
    'tanh':nn.Tanh,
    'relu':nn.ReLU
}

import numpy as np

def generate_decreasing_powers_of_2(S, E, N):
    # Generate B numbers between 1 and A using linspace
    linear_space = np.linspace(S, E, N+1)
    
    # Round these numbers to the nearest power of 2
    powers_of_2_sequence = [2**np.floor(np.log2(x)).astype(int) for x in linear_space]
    powers_of_2_sequence[0] = S
    powers_of_2_sequence[-1]= E
    return powers_of_2_sequence

class ShortCut(nn.Module):
    def __init__(self, layers, down2half):
        super().__init__()
        self.layers = layers
        self.down2half = down2half

    def forward(self, x):
        res = self.layers(x)
        #print(f"{x.shape} {res.shape} {self.down2half}")
        if self.down2half:
            x = (x[:,::2] + x[:,1::2])/2
        return x + res

class AutoDownstreamBuilder:
    skip_loss_backward_key=[]
    skip_downstream_key = []
    slide_feature_window_size = None
    isinrecurrent_mode = False
    def freeze_downstream(self, freeze_mode, mode='train'):
        if freeze_mode is None:return
        if not freeze_mode: return  
        layer_type_that_effect_by_eval = (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm, torch.nn.Dropout)
        if 'only_downstream' in freeze_mode:
            for param in self.parameters():
                param.requires_grad = False
            for module in self.modules():
                module.eval()  # Set the module to evaluation mode
            downstream_keys = freeze_mode.replace('only_downstream','').strip('.').split(',')
            for downstream_key in downstream_keys:
                print0(f"The model is freezen, except {downstream_key} branch !!! ")
                for param in self.predictor[downstream_key].parameters():
                    param.requires_grad = True
                if mode == 'train':
                    for module in self.predictor[downstream_key].modules():
                        module.train()  # Set the module to evaluation mode
            for key in self.predictor.keys():
                if key not in downstream_keys:
                    print0(f"Since you are using freeze all except downstream, lets skip  {key} branch !!! ")
                    self.skip_loss_backward_key.append(key)
        else:
            raise NotImplementedError(f"freeze mode {freeze_mode} not support")
            # do nothing
        

    def build_downstream_task(self, upstream_config:BackboneConfig, downstream_config:PredictorConfig, downstream_input_dim_size=None):
        if downstream_input_dim_size is None: downstream_input_dim_size = upstream_config.hidden_size

        downstream_pool = downstream_config.downstream_pool
        self.predictor = nn.ModuleDict()
        self.loss_pool = nn.ModuleDict()
        self.function_to_origin = {}
        for task, task_config in downstream_pool.items():
            loss_type = task_config['metric']
            pred_dims = task_config['channel']
            bias_flag = task_config['bias']
            nonlinear = task_config['nonlinear']
            num_layer = task_config.get('layers', 1)
            intermediate_nonlinear = task_config.get('intermediate_nonlinear', 'tanh')
            normlizer = task_config.get('normlizer', {})
            value_unit= task_config.get('unit',1)
            shortcut    = task_config.get('shortcut',False)
            module_type = task_config.get('module_type','linear')
            if task in ['phase']:
                self.predictor[task] = nn.Linear(downstream_input_dim_size, upstream_config.vocab_size, bias=False, module_type=module_type) # share use vocab_size. Not good!
                self.loss_pool[task] = torch.nn.CrossEntropyLoss()
            elif task in ['findP','findS']:
                #assert upstream_config.wave_channel < 4, "the wave channel should be less than 4 for findP and findS task. Which means we only input wave information"
                self.slide_feature_window_size = downstream_config.slide_feature_window_size
                self.slide_stride_in_training = downstream_config.slide_stride_in_training
                assert pred_dims == 1, "findP and findS prediction dim is fixed as 1, bias is False"
                #assert bias_flag == False, "findP and findS prediction dim is fixed as 1, bias is False"
                self.predictor[task]                = nn.ModuleDict()
                
                self.predictor[task]["judger"]      = self.build_one_downstream_branch(downstream_config, downstream_input_dim_size, 
                                                                                       pred_dims, num_layer, nonlinear, 
                                                                                       bias_flag, intermediate_nonlinear, module_type='linear')
                
                
                self.predictor[task]["distribution"]= self.build_one_downstream_branch(downstream_config, downstream_input_dim_size,
                                                                                        pred_dims, num_layer, nonlinear, 
                                                                                        bias_flag, intermediate_nonlinear, module_type=module_type, shortcut=shortcut)
                self.loss_pool[task] = DistributionPosition(**task_config.get('criterion_config',{}))
                self.function_to_origin[task] = get_normlizer_convert()
            # elif task in ['hasP']:
            #     self.predictor[task] = nn.Linear(downstream_input_dim_size, 2, bias=False)
            #     self.loss_pool[task] = AdaptCrossEntropyLoss()
            # elif task in ['findP']:
            #     self.predictor[task] = nn.Linear(downstream_input_dim_size, 1, bias=False)
            #     self.loss_pool[task] = KLDivPosition()
            else:
                ### build loss function
                if loss_type == 'uncertainty_loss':
                    # weight = torch.sigmoid((torch.arange(upstream_config.context_length)-upstream_config.context_length//2)/100)
                    # weight = weight/weight.sum()        
                    # weight = weight[self.min_used_length:]   
                    self.loss_pool[task] = UncertaintyLoss((upstream_config.context_length - self.min_used_length,))
                elif loss_type == 'fastABS':
                    self.loss_pool[task] = FastABS()
                elif loss_type == 'MSE':
                    self.loss_pool[task] = BroadCastMSE()
                elif loss_type == 'cosine':
                    self.loss_pool[task] = CosineLoss()
                elif loss_type == 'cosinesimilirity':
                    self.loss_pool[task] = CosineSimilirity()
                elif loss_type == 'parallelQsimilirity':
                    self.loss_pool[task] = ParallelQSimilirity()
                elif loss_type == 'antil':
                    self.loss_pool[task] = AntiLoss()
                elif loss_type == 'adace':
                    self.loss_pool[task] = AdaptCrossEntropyLoss()
                elif loss_type == 'klpos':
                    self.loss_pool[task] = KLDivPosition()
                elif loss_type == 'dbpos':
                    self.loss_pool[task] = DistributionPosition(**task_config.get('criterion_config',{}))
                elif loss_type == 'focal':
                    self.loss_pool[task] = FocalLoss(alpha=downstream_config.focal_loss_alpha, gamma=downstream_config.focal_loss_gamma)
                elif loss_type == 'bce':
                    self.loss_pool[task] = torch.nn.BCEWithLogitsLoss()
                elif loss_type == 'ce':
                    self.loss_pool[task] = torch.nn.CrossEntropyLoss()
                elif loss_type == 'vector_distance':
                    self.loss_pool[task] = VectorDistance()
                else:
                    raise NotImplementedError

                
                
                assert value_unit == 1, "After 20231212, the normlizer args should be assigned. Thus, please use unit flag in the normlizer"
                if not isinstance(normlizer, _NormlizerConvert):
                    assert isinstance(normlizer, dict), "normlizer must be a dict"
                    normlizer = get_normlizer_convert(**normlizer)
                self.function_to_origin[task] = normlizer

                ### build downstreambranch
                if num_layer ==0: 
                    print0(f'====> Skip build {task} downstream branch because num_layer==0')
                    continue
                if loss_type in ['uncertainty_loss']:pred_dims*=2
                self.predictor[task] = self.build_one_downstream_branch(downstream_config, downstream_input_dim_size, pred_dims, num_layer, 
                                                                        nonlinear, bias_flag, intermediate_nonlinear, module_type=module_type)

    def build_one_downstream_branch(self,downstream_config,downstream_input_dim_size, pred_dims, num_layer, 
                                    nonlinear, bias_flag, intermediate_nonlinear, module_type='linear', shortcut=False):
        layer = nn.Sequential()
        if downstream_config.normlize_at_downstream:#<--- only for retnet, be careful the eps!!! it is supposed not to change
            layer.append(RMSNorm(downstream_input_dim_size, eps=1e-6))
        
        dim_sequence = generate_decreasing_powers_of_2(downstream_input_dim_size, pred_dims, num_layer)
        if module_type == 'cnn':
            layer.append(Rearrange('B L C-> B C L'))

        for i in range(num_layer):
            now_layer = []
            if downstream_config.downstream_dropout: 
                now_layer.append(nn.Dropout(downstream_config.downstream_dropout))
            if module_type == 'linear':
                projection_layer = nn.Linear(dim_sequence[i], dim_sequence[i+1], bias=bias_flag)
            elif module_type in ['cnn','cnn2']:
                if module_type == 'cnn2': #### <--- cnn2 always fail, will remove in the future
                    projection_layer = nn.Sequential(
                        Rearrange('B L C-> B C L'),
                        nn.Conv1d(dim_sequence[i], dim_sequence[i+1], kernel_size=3, padding=1, bias=bias_flag),
                        Rearrange('B C L -> B L C'),
                        nn.LayerNorm(dim_sequence[i+1])
                    )
                
                else:
                    projection_layer = nn.Sequential(
                        nn.Conv1d(dim_sequence[i], dim_sequence[i+1], kernel_size=3, padding=1, bias=bias_flag),
                        nn.BatchNorm1d(dim_sequence[i+1])
                    )
            else:
                raise NotImplementedError
            now_layer.append(projection_layer)
            non_linear_layer = NonLinearPool[nonlinear]() if i == num_layer - 1 else NonLinearPool[intermediate_nonlinear]()
            now_layer.append(non_linear_layer)
            if shortcut and i != num_layer - 1:
                layer.append(ShortCut(nn.Sequential(*now_layer), down2half = (dim_sequence[i]//2 == dim_sequence[i+1])  ))
            else:
                for lll in now_layer:layer.append(lll)
                #layer.append(nn.Sequential(*now_layer))
        if module_type == 'cnn':
            layer.append(Rearrange('B C L -> B L C'))
        return layer
    
    def downstream_prediction(self, downstream_feature:dict, keys=None): 
        #hidden_states is (B, 1, C) but hidden_states_full is (B, L ,C)

        keys = downstream_feature.keys() if keys is None else keys
        preded = {}
        for key in keys:
            if key in self.skip_downstream_key: continue
            layer = self.predictor[key]
            hidden_states = downstream_feature[key]
            #move below to get_hidden_state
            #if hidden_states.shape[1] > self.min_used_length:
            # hidden_states = hidden_states[:, self.min_used_length:]
            if key in ['findP','findS']:
                assert self.slide_feature_window_size is not None, "in the latest version, it is needed to provide the slide_feature_window_size"
                #assert hidden_states[1].shape[1] == self.slide_feature_window_size, f"the findP and findS must have same length as slide_feature_window_size={self.slide_feature_window_size}, but {hidden_states[1].shape[1]} != {self.slide_feature_window_size}"
                self.total_length   = hidden_states.shape[-2]
                
                hasP_token  = layer['judger'](hidden_states) #(B, L , C)->(B, L ,1), # make sure the f(g(x)) == g(f(x))
                assert hasP_token.shape[-1] == 1
                hasP_token  = F.avg_pool1d(hasP_token[...,0], self.slide_feature_window_size, self.slide_stride_in_training) # (B,N, )

                whereisP_token = layer['distribution'](hidden_states) #(B, L, C) -> (B, L, 1 )
                assert whereisP_token.shape[-1] == 1 #
                whereisP_token = whereisP_token[...,0].unfold(1, self.slide_feature_window_size, self.slide_stride_in_training) # (B,N, L)
                preded[key] = torch.cat([hasP_token.unsqueeze(-1), whereisP_token],-1) #(B, N , L+1)
                
            elif key in ['ESWN','SPIN']:
                preded[key] = layer(hidden_states).mean(1, keepdims=True)
            else:
                preded[key] = layer(hidden_states)

        return preded
       
    def get_confidence(self, confidence):
        assert self.config.Predictor.use_confidence != 'whole_sequence', 'now need confidence for whole_sequence'
        if self.config.Predictor.use_confidence == 'status':
            ### the status must be ===> N0P1S2 <===
            confidence = (self.pick_up_right_token_for_stride_mode(confidence)>0).float() # (B, S)
        else:#if self.config.Predictor.use_confidence == 'P-2to30':
            ### the status must be ===> P-2to30 <===
            confidence = self.pick_up_right_token_for_stride_mode(confidence) #(self.pick_up_right_token_for_stride_mode(confidence)>0).float() # (B, S)
        return confidence.float()
  
    def evaluate_error(self,target,preded, get_prediction=None, report_error=True):
        assert len(preded) > 0, f"it seem your prediction pool is empty, check your model"
        error_record = {}
        prediction   = {}
        loss         = 0

        for key in preded.keys():
            #print(f"{key} => {preded[key].shape} => {target[key].shape}")
            assert key in target, f"the target key is {target.keys()} but your preded key is {preded.keys()}"
            if key in ['phase', 'logits','status','phase_probability', 'probabilityPeak', 'P_Peak_prob','ESWN','SPIN']:                
                # shift_logits = preded[key][..., :-1, :].flatten(0, 1)
                # shift_labels = target[key][..., 1:].flatten(0, 1)  # (B,L) ->(BL) # notice we roll it at the beginning
                if key in ['phase','logits']:
                    shift_logits,shift_labels = self.build_phase_pair(preded[key], target[key], flatten=False)
                elif key in ['status']:
                    # print(preded[key].shape)
                    # print(target[key].shape)
                    shift_logits,shift_labels = self.build_status_pair(preded[key], target[key].long(), flatten=False)
                    # print(shift_logits.shape)
                    # print(shift_labels.shape)
                    # raise
                elif key in ['phase_probability','probabilityPeak']:
                    shift_logits,shift_labels = self.build_status_pair(preded[key], target[key], flatten=False)
                elif key in ['P_Peak_prob']:
                    shift_logits,shift_labels = self.build_status_pair(preded[key][...,0], target[key], flatten=False)
                elif key in ['ESWN','SPIN']:
                    shift_logits,shift_labels = preded[key], target[key].long()
                else:
                    raise NotImplementedError

                
                now_loss = self.loss_pool[key](shift_logits.flatten(0,1), shift_labels.flatten(0,1))
                error_record[f'loss_{key[:3]}'] = now_loss.item()
                loss += now_loss
                if report_error:
                    if get_prediction is not None:
                        if key in ['phase_probability', 'probabilityPeak', 'ESWN','SPIN']:
                            prediction[key] = {'pred': torch.softmax(shift_logits.float(),-1).detach().cpu().numpy()} # <-- return the probability
                        elif key in ['P_Peak_prob']:
                            prediction[key] = {'pred': torch.sigmoid(shift_logits.float()).detach().cpu().numpy()} # <-- return the probability
                        else:
                            pass
                        if 'real' in get_prediction and key in prediction:
                            prediction[key]['real'] = shift_labels.float().detach().cpu().numpy()
                    _, predicted_classes = torch.max(shift_logits, -1)
                    if key in ['phase_probability', 'probabilityPeak']:
                        _, shift_labels= torch.max(shift_labels, -1)
                        accuracy = (predicted_classes == shift_labels).float().mean()
                        #accuracy = correct / np.prod(shift_labels.shape[0:-1])
                        error_record[f'a_{key}'] = accuracy.float().detach().item()

                    if key in ['ESWN','SPIN']:
                        accuracy = (predicted_classes == shift_labels).float().mean()
                        #accuracy = correct / np.prod(shift_labels.shape[0:-1])
                        error_record[f'a_{key}'] = accuracy.float().detach().item()
            else:
                pred  = preded[key] # (B, L, 1) or (B, L, 2) <-- the first dimension is the prediction
                real  = target[key] # (B, L) or (B,1)

                if self.config.Predictor.use_confidence != 'whole_sequence': ### use_confidence == 'status'
                    
                    confidence = self.get_confidence(target['confidence'])
                    now_loss   = self.loss_pool[key](pred.squeeze(-1), real, weight=confidence)
                else:
                    now_loss = self.loss_pool[key](pred.squeeze(-1), real)
                if key not in self.skip_loss_backward_key: loss += now_loss
                error_record[f'loss_{key[:3]}'] = now_loss.item()
                if report_error:
                    pred = pred
                    real = real
                    if get_prediction:
                        prediction[key] = {'pred':pred.float().detach().cpu().numpy()}
                        if 'real' in get_prediction:prediction[key]['real'] = real.float().detach().cpu().numpy()
                    if key in ['angle', 'line']:
                        pass
                    elif key in ['findP','findS']:
                        if len(pred.shape)==3:
                            pred = pred.flatten(0,1)# (BN,L+1)
                            real = real.flatten(0,1)# (BN,)
                        L     = pred.shape[-1]
                        real  = real.unsqueeze(-1)  #(B,L+1)
                        pred  = pred.unsqueeze(-1)  #(B,)
                        p_pos = torch.argmax(pred, dim=1) - 1 #(B,)
                        slience_index = p_pos==-1 # (B,)
                        p_pos[slience_index] = -L # make the slience case be -L and then divide by L to -1
                        p_pos = p_pos/L # silence case is -1, otherwise it is in [0,1]
                        ### <--- error in positive activate 
                        error_record[f'e_{key}'] = (p_pos - real).abs().mean().item()
                        
                        # accu_matrix = compute_accu_matrix(~slience_index, real>=0)
                        
                        # for accu_type, val in accu_matrix.items():
                        #     error_record[f'{accu_type}_P'] = val.item()
                    else:
                        #print(f"key is {key} and pred shape is {pred.shape} and real shape is {real.shape}")
                        if key not in ['group_vector','angle_vector']:
                            pred = pred[...,0]
                        
                        error_record[f'e_{key}'] = (self.function_to_origin[key].recovery_from_machine_data(pred) - 
                                                    self.function_to_origin[key].recovery_from_machine_data(real)          ).abs().mean().item()
    
        error_record_extra, prediction_extra = extra_metric_computing(preded, target, get_prediction, self.function_to_origin, report_error = report_error)
        error_record = error_record|error_record_extra
        prediction   = prediction|prediction_extra
        return loss, error_record, prediction
    
    def deal_with_labels(self, labels, preded, confidence=None):
        
        target = {'confidence':confidence}

        for key, val in labels.items():
            if key not in preded:continue
            if key in ['findP','findS']:
                #if len(preded[key])!=len(val):
                total_length = self.total_length # the total_length is the origin length (B, S, D), not the preded[key] length (B, L1, D) 
                ### use self.total_length is quite a trash coding way, but it is the only way to get the total_length

                val   = val*total_length ## (val is a float in [0,1] which means the position of the target in ratio, for example 0.8)
                ### we need make it back to the true position in samples (if we downsampling like CNN-3, then it should be also at the downsampling 0.8 and true samples is 0.8*S)
                delta = self.slide_stride_in_training
                
                num_stride_jumps = preded[key].shape[1]
                shifted_target = torch.cat([val - i*delta for i in range(num_stride_jumps)], -1)/self.slide_feature_window_size#(B,1) -> (B,N)
                
                shifted_target[shifted_target<0]=-1 ## target at left side of the window is -1
                shifted_target[shifted_target>1]=-1 ## target at right side of the window is -1
                target[key] = shifted_target
            elif len(val.shape) > 1 and self.min_used_length and val.shape[-1] > self.min_used_length:
                target[key] = val[..., self.min_used_length:]
            elif len(val.shape) == 1:
                target[key] = val.unsqueeze(1)  # ==> (B,1) ==> (B,L)
                if preded[key].shape[1] > 1:
                    target[key] = target[key].repeat(1, preded[key].shape[1])
            elif len(val.shape)==2 and val.shape[-1]==2 and (len(val.shape) == len(preded[key].shape)-1):
                target[key] = val.unsqueeze(1)  # ==> (B,1) ==> (B,L)
                target[key] = target[key].repeat(1, preded[key].shape[1],1)
            else:
                target[key] = val
         
        return target
    
    def compute_loss(self, preded, labels, get_prediction=None):
        target = self.deal_with_labels(labels, preded)
        loss, error_record, prediction = self.evaluate_error(target, preded, get_prediction=get_prediction)
        return loss, error_record, prediction

def extra_metric_computing(preded, target, get_prediction, function_to_origin:Dict[str, _NormlizerConvert]=None, report_error = True):
    error_record = {}
    prediction   = {}
    assert function_to_origin is not None, "function_to_origin must be provided"
    for key, val in preded.items():
        engine = 'torch' if isinstance(val, torch.Tensor) else 'numpy'
    if 'distance' in preded and 'angle' in preded and report_error:
        key =  'shift'
        a     = function_to_origin['distance'].recovery_from_machine_data(target['distance'])
        b     = function_to_origin['distance'].recovery_from_machine_data(preded['distance'][...,0])
        theta = (target['angle'] - preded['angle'][..., 0])*np.pi # the range of angle is [-1,1]
        if engine == 'torch':
            shift = torch.sqrt(torch.clamp(a**2 + b**2 - 2*a*b*(theta.cos()), min=0))
            shift = shift.float().detach().numpy()
            
        else:
            shift = np.sqrt(np.clip(a**2 + b**2 - 2*a*b*(theta.cos()), 0, np.inf))
        if get_prediction:
            prediction[key] = {'pred': shift}
            if 'real' in get_prediction:prediction[key]['real']= np.zeros((len(shift),1))
        error_record['e_shift'] = shift.mean()

    if 'distance' in preded and 'angle' in preded and 'deepth' in preded and report_error:
        target_deepth = function_to_origin['deepth'].recovery_from_machine_data(target['deepth'])
        preded_deepth = function_to_origin['deepth'].recovery_from_machine_data(preded['deepth'][...,0])
        delta_deepth = (target_deepth - preded_deepth)
        if engine == 'torch':
            shift = torch.sqrt(torch.clamp(a**2 + b**2 - 2*a*b*(theta.cos()), min=0))
            earth_dist   = torch.sqrt(shift**2 + delta_deepth**2).float().detach().cpu().numpy()
            earth_dist_e = earth_dist.mean().float().detach().item()
        else:
            earth_dist   = np.sqrt(shift**2 + delta_deepth**2)
            earth_dist_e = earth_dist_e.mean()
        if get_prediction:
            prediction[key] = {'pred': earth_dist}
            if 'real' in get_prediction:prediction[key]['real'] = np.zeros((len(earth_dist),1))
        error_record['earth_dist'] = earth_dist_e
        
    if 'x' in preded and 'y' in preded and report_error:
        target_x = function_to_origin['x'].recovery_from_machine_data(target['x'])
        target_y = function_to_origin['y'].recovery_from_machine_data(target['y'])
        pred_x   = function_to_origin['x'].recovery_from_machine_data(preded['x'][...,0])
        pred_y   = function_to_origin['y'].recovery_from_machine_data(preded['y'][...,0])
        if engine == 'torch':
            location_origin = torch.stack([target_x, target_y],-1) #(B, L , 2)
            location_preded = torch.stack([  pred_x,   pred_y],-1) #(B, 1 , 2)
            location_origin_norm = location_origin.norm(dim=-1)
            location_preded_norm = location_preded.norm(dim=-1)
            distance_error  = torch.mean(torch.abs(location_origin_norm - location_preded_norm)).item()
            shift           = (location_origin - location_preded).norm(dim=-1)
            absolute_error  = torch.mean(shift).item()
            
        else:
            
            location_origin = np.stack([target_x, target_y],-1) #(B, L , 2)
            location_preded = np.stack([  pred_x,   pred_y],-1) #(B, 1 , 2)
            location_origin_norm = np.linalg.norm(location_origin, axis=-1)
            location_preded_norm = np.linalg.norm(location_preded, axis=-1)
            distance_error  = np.mean(np.abs(location_origin_norm - location_preded_norm))
            shift           = np.linalg.norm(location_origin - location_preded, axis=-1)
            absolute_error  = np.mean(shift)
        
        if get_prediction:
            distance = location_preded_norm.float().detach().cpu().numpy() if engine == 'torch' else location_preded_norm
            prediction['distance'] = {'pred': distance}
            if 'real' in get_prediction:
                prediction['distance']['real'] = location_origin_norm.float().detach().cpu().numpy()  if engine == 'torch' else location_origin_norm
            shift = shift.float().detach().cpu().numpy() if engine == 'torch' else shift
            prediction['shift'] = {'pred': shift}
            if 'real' in get_prediction:
                prediction['shift']['real'] = np.zeros((len(shift),1))
            if engine == 'torch':
                target_x = target_x.float().detach().cpu().numpy() 
                target_y = target_y.float().detach().cpu().numpy() 
                pred_x   = pred_x.float().detach().cpu().numpy()   
                pred_y   = pred_y.float().detach().cpu().numpy()       
            target_angle_deg = np.rad2deg(np.arctan2(target_y, target_x))
            pred_angle_deg   = np.rad2deg(np.arctan2(pred_y, pred_x))
            prediction['pred_angle_deg'] = {'pred': pred_angle_deg,
                                            'real': target_angle_deg,
                                   }
        error_record['e_distance'] = distance_error
        error_record['e_shift'] = absolute_error
    
    if 'deepth' in preded and 'x' in preded and 'y' in preded and report_error:
        target_x = function_to_origin['x'].recovery_from_machine_data(target['x'])
        target_y = function_to_origin['y'].recovery_from_machine_data(target['y'])
        target_deepth = function_to_origin['deepth'].recovery_from_machine_data(target['deepth'])
        pred_x   = function_to_origin['x'].recovery_from_machine_data(preded['x'][...,0])
        pred_y   = function_to_origin['y'].recovery_from_machine_data(preded['y'][...,0])
        pred_deepth = function_to_origin['deepth'].recovery_from_machine_data(preded['deepth'][...,0])
        if engine == 'torch':
            location_origin = torch.stack([target_x, target_y, target_deepth], -1)
            location_preded = torch.stack([pred_x, pred_y, pred_deepth], -1)
            earth_dist = (location_origin - location_preded).norm(dim=-1).float().detach().cpu().numpy()
        else:
            location_origin = np.stack([target_x, target_y, target_deepth], -1)
            location_preded = np.stack([pred_x, pred_y, pred_deepth], -1)
            earth_dist      = np.linalg.norm(location_origin - location_preded, axis=-1)
        if get_prediction:
            prediction['earth_dist'] = {'pred': earth_dist}
            if 'real' in get_prediction:prediction['earth_dist']['real'] = np.zeros((len(earth_dist),1))
        earth_dist = np.mean(earth_dist)
        error_record['earth_dist'] = earth_dist

    for key in ['line_vector', 'angle_vector']:
        if not (key in preded and report_error):continue
        target_angle_vector = function_to_origin[key].recovery_from_machine_data(target[key])
        pred_angle_vector   = function_to_origin[key].recovery_from_machine_data(preded[key])
        if engine == 'torch':
            target_angle_vector = target_angle_vector.squeeze()
            pred_angle_vector   = pred_angle_vector.squeeze()
            assert len(target_angle_vector.shape) == len(pred_angle_vector.shape), f"target_angle_vector.shape {target_angle_vector.shape} != pred_angle_vector.shape {pred_angle_vector.shape}"
            target_angle_vector = target_angle_vector/target_angle_vector.norm(dim=-1, keepdim=True)
            pred_angle_vector   = pred_angle_vector/pred_angle_vector.norm(dim=-1, keepdim=True)
            target_angle_vector = target_angle_vector.float().detach().cpu().numpy()
            pred_angle_vector   = pred_angle_vector.float().detach().cpu().numpy()
        else:
            target_angle_vector = np.squeeze(target_angle_vector)
            pred_angle_vector   = np.squeeze(pred_angle_vector  )
            assert len(target_angle_vector.shape) == len(pred_angle_vector.shape), f"target_angle_vector.shape {target_angle_vector.shape} != pred_angle_vector.shape {pred_angle_vector.shape}"
            target_angle_vector = target_angle_vector/np.linalg.norm(target_angle_vector, axis=-1, keepdims=True)
            pred_angle_vector   = pred_angle_vector/np.linalg.norm(pred_angle_vector, axis=-1, keepdims=True)
        angle_shift_error   = np.linalg.norm(target_angle_vector - pred_angle_vector, axis=-1)
        angle_x_error       = np.abs(target_angle_vector[...,0] - pred_angle_vector[...,0])
        angle_y_error       = np.abs(target_angle_vector[...,1] - pred_angle_vector[...,1])
        target_angle_deg    = np.rad2deg(np.arctan2(target_angle_vector[..., 1], target_angle_vector[..., 0]))
        pred_angle_deg      = np.rad2deg(np.arctan2(pred_angle_vector[..., 1], pred_angle_vector[..., 0]))
        if key in ['line_vector']:
            target_angle_deg= target_angle_deg/2
            pred_angle_deg  = pred_angle_deg  /2
            delta_angle    = np.abs(target_angle_deg - pred_angle_deg)%90
        if key in ['angle_vector']:
            
            delta_angle    = np.abs(target_angle_deg - pred_angle_deg)%180
            
        if get_prediction:
            prediction['delta_angle'] = {'pred': delta_angle,
                                         'real': np.zeros_like(delta_angle),
                                   }
            prediction['pred_angle_deg'] = {'pred': pred_angle_deg,
                                            'real': target_angle_deg,
                                   }


        angle_shift_error = np.mean(angle_shift_error)
        angle_x_error     = np.mean(angle_x_error)
        angle_y_error     = np.mean(angle_y_error)
        angle_delta_angle = np.mean(delta_angle)
        error_record['record_angle_shift'] = angle_shift_error
        error_record['record_angle_x'] = angle_x_error
        error_record['record_angle_y'] = angle_y_error
        error_record['delta_degree'] = angle_delta_angle
    
    if 'group_vector' in preded:
        target_group_vector = function_to_origin['group_vector'].recovery_from_machine_data(target['group_vector'])#(B, L , 2)
        pred_group_vector   = function_to_origin['group_vector'].recovery_from_machine_data(preded['group_vector'])#(B, 1 , 2)
        if engine == 'torch':
            location_origin = target_group_vector #(B, L , 2)
            location_preded = pred_group_vector   #(B, 1 , 2)
            shift           = (location_origin - location_preded).norm(dim=-1)
            absolute_error  = torch.mean(shift).item()
        else:
            location_origin = target_group_vector #(B, L , 2)
            location_preded = pred_group_vector   #(B, 1 , 2)
            shift           = np.linalg.norm(location_origin - location_preded, axis=-1)
            absolute_error  = np.mean(shift)
        if get_prediction:
            shift = shift.float().detach().cpu().numpy() if engine == 'torch' else shift
            key = 'group_shift'
            prediction[key] = {'pred': shift}
            
            if 'real' in get_prediction:prediction[key]['real'] = np.zeros((len(shift),1))
        error_record['group_shift']    = absolute_error

        if 'station_mask' in target:
            group_splitlist = target['station_mask']
            delta_vector = target_group_vector - pred_group_vector
            if engine == 'torch':
                delta_vector = delta_vector.float().detach().cpu().numpy()
            group_shifts = []
            offset = 0
            for row_id, row in enumerate(group_splitlist):
                element_num = len(row[row>0])
                elements    = delta_vector[offset:offset+element_num]
                
                elements    = np.mean(elements,axis=0)
                group_shifts.append(np.linalg.norm(elements,axis=-1))
                offset += element_num
            assert offset == len(delta_vector), f"given a splitlist for total length {target_group_vector.shape} but the sum of splitlist is {offset}"
            group_shifts = np.stack(group_shifts) #(N,L)
        error_record['GM_shift']    = group_shifts.mean()
        if get_prediction:
            key = 'GM_shift'
            prediction[key] = {'pred': group_shifts}
            if 'real' in get_prediction:prediction[key]['real'] = np.zeros((len(group_shifts),1))
    return error_record, prediction