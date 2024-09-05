import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Dict, Optional, List,Tuple, Union
from .retnet.modeling_retnet import RetNetModel, RetNetOutputWithPast, PreTrainedModel, RetNetConfig, UnitLayerNorm
from .backbone_config import RetnetConfig
from ..SignalModel import SignalModelBuilder
from ...predictor.DownstremBuilder import AutoDownstreamBuilder
from ...embedding.EmbedderBuilder import AutoEmbedderBuilder
from ...model_arguements import SignalModelConfig, FreezeConfig
from trace_utils import print0
import os

@dataclass
class SignalOutput(RetNetOutputWithPast):

    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    prediction: Optional[Dict[str, torch.Tensor]] = None
    error_record: Optional[Dict[str, torch.Tensor]] = None
    state: Optional[List[Dict[str, torch.FloatTensor]]] = None
 
class RetNetForSignalBase(PreTrainedModel, SignalModelBuilder, AutoEmbedderBuilder, AutoDownstreamBuilder):
    _tied_weights_keys = []
    min_used_length = 0
    config_class = SignalModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["backbone.RetNetDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def __init__(self, args: SignalModelConfig):
        super().__init__(args)
        self.min_used_length = 0
        self.build_signal_embedder(args)
        self.build_backbone(args.Backbone)
        self.build_downstream_task(args.Backbone, args.Predictor)
        self.post_init()
        
    def freeze_model_during_train(self, freeze_config:Optional[FreezeConfig]=None):
        if freeze_config is not None:
            self.freeze_embedder(freeze_config.freeze_embedder)
            self.freeze_backbone(freeze_config.freeze_backbone)
            self.freeze_downstream(freeze_config.freeze_downstream)

    def build_downstream(self, args: SignalModelConfig):
        input_dim_size = args.Backbone.hidden_size*args.Backbone.num_hidden_layers if args.use_whole_layer_output else args.Backbone.hidden_size
        self.build_downstream_task(args.Backbone, args.Predictor, 
                                   downstream_input_dim_size=input_dim_size)
    
    def formalize_hidden_state(self, outputs:RetNetOutputWithPast, all_feature=False):
        if all_feature: return outputs.hidden_states
        if self.config.use_whole_layer_output:
            return torch.cat(outputs.hidden_states,-1)
        else:
            return outputs.last_hidden_state
        
    def deal_with_autoregress_sequence(self, status_seq):
        return status_seq
    
    def build_signal_embedder(self, args):
        
        self.build_wave_embeder(args.Embedding)
     
    def kernel_forward(self, inputs_embeds, past_kv=None, use_cache=False, forward_impl='parallel')->Tuple[RetNetOutputWithPast, Dict[str, torch.Tensor]]:
        outputs = self.get_kernel_output(inputs_embeds, past_kv, use_cache, forward_impl)
        return self.collect_kernel_output(outputs)

    def get_kernel_output(self, inputs_embeds, past_kv=None, use_cache=False, forward_impl='parallel', fixed_seq_len=None)->RetNetOutputWithPast:
        outputs = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_kv,
            use_cache=use_cache,
            forward_impl=forward_impl, 
            fixed_seq_len=fixed_seq_len,
            output_hidden_states=self.config.use_whole_layer_output
        )
        return outputs

    def collect_kernel_output(self, outputs):
        raise NotImplementedError
       
    @staticmethod
    def build_backbone_config(args: RetnetConfig):
        config = RetNetConfig(decoder_embed_dim      =args.hidden_size,
                              decoder_value_embed_dim=args.intermediate_size,
                              decoder_ffn_embed_dim  =args.attention_hidden_size,
                              decoder_layers         =args.num_hidden_layers,
                              decoder_retention_heads=args.num_heads,
                              use_flash_retention=args.use_flash_retention,
                              normlize_for_stable=args.normlize_for_stable,
                              disable_all_bias   =args.disable_all_bias,
                              use_lm_decay       =args.use_lm_decay,
                              normalize_at_end   =args.normalize_at_end,
                              retention_mode       =args.retention_mode,
                              dropout = args.retention_dropout,
                              activation_dropout = args.retention_dropout,
                              monitor_retention = args.monitor_retention,
                              groupnorm_type=args.groupnorm_type,
                              #decoder_normalize_before   = args.normalize_before_retention,

                              )
 
        return config
    
    def build_backbone(self, args):
        config  = self.build_backbone_config(args)
        self.backbone= RetNetModel(config)
        self.backbone.embed_tokens = None

    def get_downstream_prediction(self, downstream, **kargs):
        return self.downstream_prediction(downstream)

    def forward(
        self,
        status_seq: Optional[torch.LongTensor] = None,
        waveform_seq: Optional[torch.FloatTensor] = None,
        trend_seq: Optional[torch.FloatTensor] = None,
        past_kv: Optional[List[torch.FloatTensor]] = None,
        labels=None,
        get_prediction: Optional[str] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        #assert input_ids is None, f"disable input_ids, please provide inputs_embeds"
        # input_ids is the type of the signal
        assert waveform_seq is not None, f"disable input_ids, so must provide inputs_embeds"

        ## the goal is use Last_Status + Now_Wave -> Now Status
        ## it seem the CUDA version requires the sequence start from 0 which is the eos_token_id,
        ## Thus we need add 1 for all the status code and make the vocob_size plus 1
        ## lets shift the status_seq
        assert status_seq is None or len(status_seq.shape) == 2
        #return_dict                = return_dict if return_dict is not None else self.config.use_return_dict
        
        status_seq          = self.deal_with_autoregress_sequence(status_seq)
        inputs_embeds       = self.get_composed_input_embedding(status_seq, waveform_seq,trend_seq=trend_seq)
        inputs_embeds       = self.get_key_token_in_parallel_mode(inputs_embeds) 
        hidden_state,downstream  = self.kernel_forward(inputs_embeds, past_kv=past_kv, use_cache=False, forward_impl='parallel')
        preded             = self.get_downstream_prediction(downstream)
        if labels is None:
            return SignalOutput(last_hidden_state=hidden_state, prediction=preded)
        else:
            if 'phase' in preded:labels['phase'] = status_seq  
            target = self.deal_with_labels(labels, preded, confidence=status_seq)
            loss, error_record, prediction = self.evaluate_error(target, preded, get_prediction=get_prediction, report_error = self.config.report_error)
            return SignalOutput(
                loss=loss,
                last_hidden_state=hidden_state,
                error_record=error_record,
                prediction=prediction
            )
        
    @torch.inference_mode()
    def generate_next(self, last_state, now_status_seq, 
                                        now_waveform_seq,
                                        now_trend_seq=None):
        self.isinrecurrent_mode = True
        assert now_status_seq.shape[1] % 3 == 0, "the length of the sequence should be multiple of 3"
        last_sequence = last_state['cached_sequence']
        last_seq_len  = last_state['seq_len']
        past_kv       = last_state['past_kv']
        if 'unnormlized_kv' in past_kv: assert past_kv['unnormlized_kv'] is not None, "you may use model.eval to enable the cache"
        if 'prev_key_value' in past_kv: assert past_kv['prev_key_value'] is not None, "you may use model.eval to enable the cache"

        extra_status_seq    = torch.cat([last_sequence['status_seq']  ,now_status_seq],1) if last_sequence['status_seq'] is not None else now_status_seq
        extra_waveform_seq  = torch.cat([last_sequence['waveform_seq'],now_waveform_seq],1) if last_sequence['waveform_seq'] is not None else now_waveform_seq
        extra_trend_seq     = torch.cat([last_sequence['trend_seq']   ,now_trend_seq],1) if last_sequence['trend_seq'] is not None else now_trend_seq

        extra_status_seq    = self.deal_with_autoregress_sequence(extra_status_seq)
        extra_inputs_embeds = self.get_composed_input_embedding(extra_status_seq,  extra_waveform_seq, trend_seq=extra_trend_seq)
        extra_inputs_embeds = self.get_key_token_in_recurrent_mode(extra_inputs_embeds)
        #extra_inputs_embeds = extra_inputs_embeds[:,4:-4]
        now_seq_len = last_seq_len + extra_inputs_embeds.shape[1]
        out = self.get_kernel_output(inputs_embeds=extra_inputs_embeds,past_kv = past_kv,use_cache=True,
                                     forward_impl='chunkwise_recurrent',
                                     fixed_seq_len= now_seq_len)
        past_kv            = out.past_key_values
        _,downstream_feature = self.collect_kernel_output(out)

        last_state = {
            'cached_sequence':{
                'status_seq'  : self.get_cached_sequence(extra_status_seq),
                'waveform_seq': self.get_cached_sequence(extra_waveform_seq),
                'trend_seq':    self.get_cached_sequence(extra_trend_seq),
            },
            'past_kv': past_kv,
            'seq_len': now_seq_len,
            'monitor': out.retentions
        }
        return last_state, out.last_hidden_state, downstream_feature

    @torch.inference_mode()
    def generate_start_pointing(self, start_status_seq, start_waveform_seq, start_trend_seq=None):
        self.isinrecurrent_mode = True
        start_status_seq    = self.deal_with_autoregress_sequence(start_status_seq)
        start_inputs_embeds = self.get_composed_input_embedding(start_status_seq, start_waveform_seq, trend_seq=start_trend_seq)
        start_inputs_embeds = self.get_key_token_in_parallel_mode(start_inputs_embeds)  
        #start_inputs_embeds = extra_inputs_embeds[:,:-4] ### <--- start index is different, the CNN use zero padding, thus fullfill the design
        start_output = self.get_kernel_output(inputs_embeds=start_inputs_embeds,past_kv = None,use_cache=True,forward_impl='chunkwise_recurrent')
        _, downstream_feature = self.collect_kernel_output(start_output) 
        start_cache = start_output.past_key_values
        assert start_cache[0] is not None, "you may use model.eval to enable the cache"
        last_state = {
            'cached_sequence':{
                'status_seq'  : self.get_cached_sequence(start_status_seq),
                'waveform_seq': self.get_cached_sequence(start_waveform_seq),
                'trend_seq':    self.get_cached_sequence(start_trend_seq),
            }, 
            'past_kv': start_cache,
            'seq_len': start_inputs_embeds.shape[1],
            'monitor': start_output.retentions
        }
        return last_state, start_output.last_hidden_state, downstream_feature
    
    @staticmethod
    def transfer_EndLayerNorm_to_Downstream_for_old_version_checkpoint(weight):
        layernorm_key    = 'backbone.layer_norm.weight'
        predictor_key = [k for k in weight.keys() if 'predictor' in k]
        new_weight = {}
        for key, val in weight.items():
            if key not in [layernorm_key] + predictor_key:
                new_weight[key] = val
                continue
            if key in predictor_key:
                sequence_name = key.split('.')
                position_in_sequence = int(sequence_name[-2])
                next_position_name = '.'.join(sequence_name[:-2]) + f".{position_in_sequence+1}." + sequence_name[-1]
                new_weight[next_position_name] = val
                if position_in_sequence == 0:
                    new_weight[key] = weight[layernorm_key]
        return new_weight

    def load_state_dict(self, state_dict, strict=True):
        if self.config.Predictor.normlize_at_downstream and 'backbone.layer_norm.weight' in state_dict:
            assert not self.backbone.config.normalize_at_end, "the model should be normalize_at_end = False"
            print0(""" 
                    Warning: You are using a model that without end layer norm but the checkpoint has end layer norm, 
                    I will assume you are tring load a args.normlize_at_downstream = True model from old checkpoints.
                    I will move the end layer norm from the backbone to each downstream task.""")
            state_dict = self.transfer_EndLayerNorm_to_Downstream_for_old_version_checkpoint(state_dict)
        super().load_state_dict(state_dict, strict=strict)

    def _init_weights(self, module):
        """
        Following original retnet, weights are already initialized in their own
        ways within their own init.
        """
        #pass
        # below is copied from LlamaPretrainedModel
        if self.config.Backbone.num_hidden_layers>4:
            print0("WARNING: for layer num > 4, we will use another initialization way. Because origin initial point will fall in NaN")
            return
        std = 0.02#self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """

        if self.config.Backbone.num_hidden_layers>4:
            print0("WARNING: for layer num > 4, we will use another initialization way. Because origin initial point will fall in NaN")
            return
            # Initialize weights
        self.apply(self._initialize_weights)

        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        self.tie_weights()

    def _set_gradient_checkpointing(self, module, value=False):
        pass


###### Directly Prediction Model
class RetNetSlidePred(RetNetForSignalBase):
    '''
    If use this mode, the sequence lenght should be fixed since we use CNN. 
    The output for the hidden state should be (B, 1, D)
    '''
    

    @staticmethod
    def get_key_token_in_parallel_mode(inputs_embeds):
        return inputs_embeds
    

    
    def collect_kernel_output(self, outputs:RetNetOutputWithPast):
        hidden_states = self.formalize_hidden_state(outputs)
        
        if   self.config.Predictor.merge_token == 'average': fea = hidden_states.mean(1, keepdims=True)
        elif self.config.Predictor.merge_token == 'last' : fea = hidden_states[:, -1:, :]
        elif self.config.Predictor.merge_token == 'first': fea = hidden_states[:, 0:1, :]
        else:
            raise ValueError("merge_token only support average, last, first")
        
        downstream_feature = {}
        for key in self.predictor.keys():
            if key in ['findP', 'findS', 'findN']:
                downstream_feature[key] = hidden_states
            # elif key in ['ESWN','SPIN']:
            #     downstream_feature[key] = hidden_states # (B, 1000, D)
            else:
                downstream_feature[key] = fea
        return hidden_states,downstream_feature #past_key_values, fea, hidden_states
    
###### Recurrent Model
class RetNetRecurrent(RetNetForSignalBase):
    """
    This model always requires the sequence length is multiple of 3 and use the MSF=3
    """
    

    
    def collect_kernel_output(self, outputs:RetNetOutputWithPast):
        hidden_states = self.formalize_hidden_state(outputs) #(B, L, D)
        assert 'findP' not in self.predictor.keys()
        downstream_feature = {key: hidden_states for key in self.predictor.keys()}
        return hidden_states, downstream_feature #past_key_values, fea, hidden_states

    

    def build_status_pair(self, status_seq_pred, status_seq_true, flatten=True):
        # status_seq_pred # (B, reduced_L, 1) # reduced_L is the length of the sequence after the CNN and cut off the tail # for example, 3000 -> 1000 -> 1000 - 4 = 996
        # status_seq_true # sh
        # Original: a b c d e f g h i j k l m n o p q r s t u v w x y z
        # CNN1    :   b     e     h     k     n     q     t     w     z
        # CNN2    :  rb     e     h     k     n     q     t     w    nz
        status_seq_true = self.pick_up_right_token_for_stride_mode(status_seq_true)
        if flatten:
            status_seq_pred = status_seq_pred.flatten(0,1)
            status_seq_true = status_seq_true.flatten(0,1)
        return status_seq_pred, status_seq_true

import numpy as np
####### Sea Model ############
class RetNetSignalSea(RetNetRecurrent):
    skip_downstream_key = ['increment']

    def freeze_backbone(self, freeze_mode, mode='train'):
        if freeze_mode is None:return
        if not freeze_mode: return 
        if freeze_mode == 'only_last4_layer':
            for param in self.parameters():
                param.requires_grad = False
            for module in self.modules():
                module.eval()  # Set the module to evaluation mode
            free_layer = self.backbone.layers[-4:]
            print0(f"The model is freezen, except last four layer !!! ")
            for param in free_layer.parameters():
                param.requires_grad = True
            if mode == 'train':
                for module in free_layer.modules():
                    module.train()  # Set the module to evaluation mode
        
        else:
            raise NotImplementedError(f"freeze mode {freeze_mode} not support")


    def get_kernel_output(self, inputs_embeds, past_kv=None, use_cache=False, forward_impl='parallel', fixed_seq_len = None,):
        outputs = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_kv,
            use_cache=use_cache,
            forward_impl=forward_impl,
            output_increment=self.config.retention_increment_type,  # <---- add this line
            fixed_seq_len=fixed_seq_len,
            output_hidden_states=self.config.use_whole_layer_output
        )
        return outputs
      
    def collect_kernel_output(self, outputs):
        hidden_states = self.formalize_hidden_state(outputs)
        downstream_feature = {}
        for key in self.predictor.keys():
            downstream_feature[key] = hidden_states #(B, L, D)
        increment     = torch.cat([t.norm(dim=(-2,-1)) if len(t.shape)==5 else t for t in outputs.increment],1)# (B, layer_num*head_num, L)
        increment     = increment.permute(0,2,1)# (B, L, layer_num*head_num)
        downstream_feature['increment'] = increment
        return hidden_states, downstream_feature
    
    

    def forward(
        self,
        status_seq: Optional[torch.LongTensor] = None,
        waveform_seq: Optional[torch.FloatTensor] = None,
        trend_seq: Optional[torch.FloatTensor] = None,
        past_kv: Optional[List[torch.FloatTensor]] = None,
        labels=None,
        get_prediction: Optional[str] = None,
    ):
        assert waveform_seq is not None, f"disable input_ids, so must provide inputs_embeds"
        assert status_seq is None or len(status_seq.shape) == 2
        
        status_seq          = self.deal_with_autoregress_sequence(status_seq)
        inputs_embeds       = self.get_composed_input_embedding(status_seq, waveform_seq,trend_seq=trend_seq)
        inputs_embeds       = self.get_key_token_in_parallel_mode(inputs_embeds) 
        hidden_state,downstream  = self.kernel_forward(inputs_embeds, past_kv=past_kv, use_cache=False, forward_impl='parallel')
        increment           = downstream.pop('increment')
        
        preded             = self.get_downstream_prediction(downstream)
        if labels is None:
            return SignalOutput(last_hidden_state=hidden_state, prediction=preded)
        else:
            if 'status' not in labels:
                #assert  status_seq.max().item() in [0,1,2], "the status_seq should be  N0P1S2 mode"
                labels['status'] = status_seq

            assert self.min_used_length == 0
            target = self.deal_with_labels(labels, preded, confidence=status_seq)
            loss, error_record, prediction = self.evaluate_error(target, preded, get_prediction=get_prediction, report_error = self.config.report_error)
            if get_prediction:
                prediction['activation'] = {}
                prediction['activation']['pred'] = increment.float().detach().cpu().numpy() ## < --- too large for save
            
            increment, true_label = self.build_status_pair(increment,labels['status'])
            activation_negtive_loss = increment[true_label == 0].mean()    # must be 0
            activation_postive_loss = (1 - increment[true_label > 0].tanh()).mean() # far away from 0
            if activation_negtive_loss > self.config.scilence_loss_threshold:
                loss = loss + self.config.scilence_alpha*activation_negtive_loss
            if activation_postive_loss > self.config.activate_loss_threshold:
                loss = loss + activation_postive_loss
            error_record['loss_scilence'] = activation_negtive_loss.item()
            error_record['loss_active']   = activation_postive_loss.item()
            if get_prediction and 'real' in get_prediction:
                prediction['activation']['real']  = true_label.float().detach().cpu().numpy()
            return SignalOutput(
                loss=loss,
                last_hidden_state=hidden_state,
                error_record=error_record,
                prediction=prediction
            )


class RetNetDirctSea(RetNetSignalSea):
    """
    After 20231210, below setting is redencency. Those function will be acomplished in Embedder.
      but we keep it for the old code
    """
    @staticmethod
    def get_cached_sequence(sequence):
        return None
    
    @staticmethod
    def get_key_token_in_recurrent_mode(inputs_embeds):
        #assert inputs_embeds.shape[1] == 9, "the input_embeds should be (B,9,D), you given {}".format(inputs_embeds.shape)
        return inputs_embeds ### every time we add 3 more stamp data and combine with last 24 and obtain the token list, due to the padding rules, only the 5th token is the new token    

    @staticmethod
    def get_key_token_in_parallel_mode(inputs_embeds):
        return inputs_embeds
    
    def pick_up_right_token_for_stride_mode(self, status_seq):
        return status_seq

class RetNetSignalSeaL1(RetNetSignalSea):

    def collect_kernel_output(self, outputs):
        assert self.config.use_whole_layer_output, "you must made use_whole_layer_output=True to obtain the first layer hiddenstate"
        hidden_states_all = self.formalize_hidden_state(outputs, all_feature=True)
        downstream_feature = {}
        for key in self.predictor.keys():
            if key in ['findP','findS','status']:
                downstream_feature[key] = hidden_states_all[0]
            else:
                downstream_feature[key] = hidden_states_all[-1] #(B, L, D)
        increment     = torch.cat([t.norm(dim=(-2,-1)) if len(t.shape)==5 else t for t in outputs.increment],1)# (B, layer_num*head_num, L)
        increment     = increment.permute(0,2,1)# (B, L, layer_num*head_num)
        downstream_feature['increment'] = increment
        return hidden_states_all[-1], downstream_feature

    def freeze_backbone(self, freeze_mode, mode='train'):
        if freeze_mode is None:return
        if not freeze_mode: return 
        if freeze_mode == 'freeze_first_layer':
            for param in self.parameters():
                param.requires_grad = False
            for module in self.modules():
                module.eval()  # Set the module to evaluation mode
            free_layer = self.backbone.layers[1:]
            print0(f"The model is freezen, except last four layer !!! ")
            for param in free_layer.parameters():
                param.requires_grad = True
            if mode == 'train':
                for module in free_layer.modules():
                    module.train()  # Set the module to evaluation mode
        else:
            raise NotImplementedError(f"freeze mode {freeze_mode} not support")
        
#### L1 model #######  
class RetNetSignalSeaL1Old(RetNetRecurrent):
    skip_downstream_key = ['increment']

    def post_init(self):
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()
        logs = []
        for n,p in self.wave_embedding.named_parameters():
            p.requires_grad = False
            logs.append(f'wave_embedding.{n} is freezen')
        if self.backbone.layer_norm is not None:
            for n,p in self.backbone.layer_norm.named_parameters():
                p.requires_grad = False
                logs.append(f'global_layer_norm.{n} is freezen') ## <-- better disable in the future version
        for n,p in self.backbone.layers[0].named_parameters():
            p.requires_grad = False
            logs.append(f'layers.0.{n} is freezen') ## <-- better disable in the future version
        if 'status' in self.predictor.keys():
            for n,p in self.predictor['status'].named_parameters():
                p.requires_grad = False
                logs.append(f'predictor.status.{n} is freezen') ## <-- better disable in the future version
        local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        if local_rank == 0:
            print("="*20 + "\n" + "\n".join(logs) + "\n" +  "="*20)

    def get_kernel_output(self, inputs_embeds, past_kv=None, use_cache=False, forward_impl='parallel', fixed_seq_len = None):
        outputs = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_kv,
            use_cache=use_cache,
            output_hidden_states=True, # <-- we need the hidden states, this must be True #self.config.use_whole_layer_output
            forward_impl=forward_impl,
            output_increment=False, # <-- we do not need the increment,
            fixed_seq_len = fixed_seq_len
        )
        return outputs
    
    def collect_kernel_output(self, outputs):
        hidden_states = self.formalize_hidden_state(outputs)
        first_hidden_state  = outputs.hidden_states[1] ## slot 0 is the inputs_embeds and slot 1 is the first hidden state
        downstream_feature = {key: hidden_states for key in self.predictor.keys() if key not in ['status']}
        if not self.training:
            downstream_feature['status'] = first_hidden_state
        return hidden_states, downstream_feature


####### Lake Model ############
class RetNetSignalLake(RetNetSignalSea):
    skip_downstream_key = ['increment', 'k_score']
    # we may not use increment for training anymore, just monitor it

    def get_kernel_output(self, inputs_embeds, past_kv=None, use_cache=False, forward_impl='parallel', fixed_seq_len = None,):
        outputs = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_kv,
            use_cache=use_cache,
            forward_impl=forward_impl,
            output_increment=self.config.retention_increment_type,  # <---- add this line, 
            fixed_seq_len=fixed_seq_len,
            output_hidden_states=self.config.use_whole_layer_output,
            output_k_score_in_column_space=self.config.output_k_score_in_column_space,
        )
        return outputs
    
    
    
    def collect_kernel_output(self, outputs):
        hidden_states = self.formalize_hidden_state(outputs)
        downstream_feature = {key: hidden_states for key in self.predictor.keys()}
        increment     = torch.cat([t.norm(dim=(-2,-1)) if len(t.shape)==5 else t for t in outputs.increment],1)# (B, layer_num*head_num, L)
        increment     = increment.permute(0,2,1)# (B, L, layer_num*head_num)
        k_score       = torch.stack(outputs.score_activate_k,-1)# (B, L, layer_num)
        downstream_feature['increment'] = increment
        downstream_feature['k_score'] = k_score
        return hidden_states, downstream_feature
    
    def forward(
        self,
        status_seq: Optional[torch.LongTensor] = None,
        waveform_seq: Optional[torch.FloatTensor] = None,
        trend_seq: Optional[torch.FloatTensor] = None,
        past_kv: Optional[List[torch.FloatTensor]] = None,
        labels=None,
        get_prediction: Optional[str] = None,
    ):
        assert waveform_seq is not None, f"disable input_ids, so must provide inputs_embeds"
        assert status_seq is None or len(status_seq.shape) == 2
        status_seq          = self.deal_with_autoregress_sequence(status_seq)
        inputs_embeds       = self.get_composed_input_embedding(status_seq, waveform_seq,trend_seq=trend_seq)
        inputs_embeds       = self.get_key_token_in_parallel_mode(inputs_embeds) 
        hidden_state,downstream  = self.kernel_forward(inputs_embeds, past_kv=past_kv, use_cache=False, forward_impl='parallel')
        increment           = downstream.pop('increment')
        k_score             = downstream.pop('k_score')
        preded             = self.get_downstream_prediction(downstream)
        if labels is None:
            return SignalOutput(last_hidden_state=hidden_state, prediction=preded)
        else:
            if 'status' not in labels:
                assert  status_seq.max().item() in [0,1,2], "the status_seq should be under N0P1S2 mode"
                labels['status'] = status_seq
            
            assert self.min_used_length == 0
            target = self.deal_with_labels(labels, preded, confidence=status_seq)
            loss, error_record, prediction = self.evaluate_error(target, preded, get_prediction=get_prediction, report_error = self.config.report_error)
            if get_prediction:
                prediction['activation'] = {}
                prediction['activation']['pred'] = increment.float().detach().cpu().numpy() ## < --- too large for save
            
            increment, true_label = self.build_status_pair(increment,labels['status'])
            activation_negtive_loss = increment[true_label == 0].mean()    # must be 0
            activation_postive_loss = (1 - increment[true_label > 0].tanh()).mean() # far away from 0
            #loss = loss + self.config.scilence_alpha*activation_negtive_loss + activation_postive_loss #<--- we dont train this 
            error_record['loss_scilence'] = activation_negtive_loss.item()
            error_record['loss_active']   = activation_postive_loss.item()

            k_score, true_label = self.build_status_pair(k_score,labels['status'])
            true_label   = (true_label > 0).float()

            k_score_loss = (k_score - true_label.unsqueeze(1)).abs().mean()
            error_record['loss_kscore']   = k_score_loss.item()
            loss = loss +  self.config.scilence_alpha*k_score_loss


            if get_prediction and 'real' in get_prediction:
                prediction['activation']['real']  = true_label.float().detach().cpu().numpy()
            return SignalOutput(
                loss=loss,
                last_hidden_state=hidden_state,
                error_record=error_record,
                prediction=prediction
            )

