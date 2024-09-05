import torch
import os
from trace_utils import print0
from dataclasses import dataclass
from transformers import PreTrainedModel 
from ..predictor.DownstremBuilder import AutoDownstreamBuilder
from ..embedding.EmbedderBuilder import AutoEmbedderBuilder
from ..model_arguements import SignalModelConfig,FreezeConfig
from transformers.utils import ModelOutput
from typing import Callable, Dict, Optional, List,Tuple, Union

class SignalModelBuilder:

    def kernel_forward(self, inputs_embeds, state, use_cache, output_attentions, output_hidden_states, return_dict):
        raise NotImplementedError


    def save_pretrained(self, path):
        if not os.path.exists(path):os.makedirs(path)
        state_dict = self.state_dict()
        file_extension  = os.path.splitext(path)[-1]
        if '.bin' not in file_extension and '.pt' not in file_extension:
            path = path = os.path.join(path, 'checkpoints.bin')
        torch.save(state_dict, path)
          
    def freeze_backbone(self, freeze_mode, mode='train'):
        if freeze_mode is None:return
        if not freeze_mode: return  
        raise NotImplementedError(f"There is no any freezen mode for {self.__class__.__name__} ")
    

@dataclass
class SignalOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    prediction: Optional[Dict[str, torch.Tensor]] = None
    error_record: Optional[Dict[str, torch.Tensor]] = None
    state: Optional[List[Dict[str, torch.FloatTensor]]] = None

class SignalBase(PreTrainedModel, SignalModelBuilder, AutoEmbedderBuilder, AutoDownstreamBuilder):

    def __init__(self, args: SignalModelConfig):
        super().__init__(args)
        self.min_used_length = 0
        self.build_signal_embedder(args)
        self.build_backbone(args.Backbone)
        self.build_downstream_task(args.Backbone, args.Predictor)

    def build_signal_embedder(self, args: SignalModelConfig):
        self.build_wave_embeder(args.Embedding)

    def freeze_model_during_train(self, freeze_config:Optional[FreezeConfig]=None):
        if freeze_config is not None:
            self.freeze_embedder(freeze_config.freeze_embedder)
            self.freeze_backbone(freeze_config.freeze_backbone)
            self.freeze_downstream(freeze_config.freeze_downstream)

    

    @staticmethod
    def build_backbone_config(args):
        raise NotImplementedError

    
    def build_backbone(self, args):
        raise NotImplementedError
                            
    def deal_with_autoregress_sequence(self, status_seq):
        return status_seq
    
    def kernel_forward(self, inputs_embeds):
        outputs = self.get_kernel_output(inputs_embeds)
        outputs = SignalOutput(last_hidden_state=outputs)
        return outputs, self.collect_kernel_output(outputs)
    
    def forward(
        self,
        status_seq: Optional[torch.LongTensor] = None,
        waveform_seq: Optional[torch.FloatTensor] = None,
        trend_seq: Optional[torch.FloatTensor] = None,
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
        outputs,downstream  = self.kernel_forward(inputs_embeds)
        preded              = self.downstream_prediction(downstream)
        if labels is None:
            return SignalOutput(last_hidden_state=outputs.last_hidden_state, prediction=preded)
        else:
            if 'phase' in preded:labels['phase'] = status_seq  
            target = self.deal_with_labels(labels, preded, confidence=status_seq)
            loss, error_record, prediction = self.evaluate_error(target, preded, get_prediction=get_prediction)
            return SignalOutput(
                loss=loss,
                last_hidden_state=outputs.last_hidden_state,
                error_record=error_record,
                prediction=prediction
            )


    def get_kernel_output(self, x):
        raise NotImplementedError(f"You should assign the way to collect output")

    def collect_kernel_output(self, outputs):
        raise NotImplementedError(f"You should assign the way to collect output")
    
    
    @torch.inference_mode()
    def generate_next(self, *args, **kargs):
        raise NotImplementedError(f"Base Signal Model do not support generate_next so far, one possible implment is cache the sequence")
       
    @torch.inference_mode()
    def generate_start_pointing(self, start_status_seq, start_waveform_seq, start_trend_seq=None):
        raise NotImplementedError(f"Base Signal Model do not support generate_next so far, one possible implment is cache the sequence")
       
    def load_state_dict(self, state_dict, strict=True):
        if self.config.Predictor.normlize_at_downstream and 'backbone.layer_norm.weight' in state_dict:
            assert not self.backbone.config.normalize_at_end, "the model should be normalize_at_end = False"
            print0(""" 
                    Warning: You are using a model that without end layer norm but the checkpoint has end layer norm, 
                    I will assume you are tring load a args.normlize_at_downstream = True model from old checkpoints.
                    I will move the end layer norm from the backbone to each downstream task.""")
            state_dict = self.transfer_EndLayerNorm_to_Downstream_for_old_version_checkpoint(state_dict)
        super().load_state_dict(state_dict, strict=strict)

    def _set_gradient_checkpointing(self, module, value=False):
        pass



