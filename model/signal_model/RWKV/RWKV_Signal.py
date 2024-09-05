
import torch
from transformers import AutoTokenizer, RwkvConfig
from transformers.models.rwkv.configuration_rwkv import RwkvConfig
# from .utils import replace_rwkv_attn_with_faster
# replace_rwkv_attn_with_faster()
from transformers.models.rwkv.modeling_rwkv import *
from typing import Dict, Optional
from .SignalModel import SignalMagTimeDis, GrowthImportanceABS, GrowthImportanceMSE, FastABS,UncertaintyLoss
from .SignalEmbedding import MultiScalerFeature
import numpy as np
@dataclass
class SignalOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    state: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    error_record: Optional[Dict[str, torch.Tensor]] = None
    prediction: Optional[Dict[str, torch.Tensor]] = None


class RwkvForSignalBase(RwkvPreTrainedModel,SignalMagTimeDis):
    _tied_weights_keys = []
    ### <----- If you change the tied weight, should delete the rwkv file in ~/.cache/torch_extension
    

    def deal_with_autoregress_sequence(self,status_seq):
        status_seq     = torch.nn.functional.pad(status_seq,(1,0))[:,:-1].long() #(B, L) -> (B, L+1) -> (B, L)
        return status_seq
    
    def get_composed_input_embedding(self, status_seq, waveform_seq):
        type_embedding = self.rwkv.embeddings(status_seq)
        wave_embedding = self.wave_embedding(waveform_seq)
        inputs_embeds  = self.embedding_merge(torch.concatenate([type_embedding, wave_embedding], -1))
        return inputs_embeds
    
    def prepare_inputs_for_generation(self, input_ids=None, state=None, inputs_embeds=None, **kwargs):
        raise
        assert input_ids is None, f"disable input_ids, please provide inputs_embeds"
        assert inputs_embeds is not None, f"disable input_ids, so must provide inputs_embeds"
        # only last token for inputs_ids if the state is passed along.
        if state is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and state is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["state"] = state
        return model_inputs
    
    def kernel_forward(self, inputs_embeds,state,use_cache,output_attentions,output_hidden_states,return_dict):
        rwkv_outputs = self.rwkv(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = rwkv_outputs[0]
        return rwkv_outputs, hidden_states
        
    def forward(
        self,
        status_seq: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        waveform_seq: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        labels=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        get_prediction: Optional[bool] = None,
    ) -> Union[Tuple, RwkvCausalLMOutput]:
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
        assert status_seq is None or len(status_seq.shape)==2
        return_dict   = return_dict if return_dict is not None else self.config.use_return_dict
        status_seq    = self.deal_with_autoregress_sequence(status_seq)
        inputs_embeds = self.get_composed_input_embedding(status_seq, waveform_seq)
        rwkv_outputs,hidden_states  = self.kernel_forward(inputs_embeds,state,use_cache,output_attentions,output_hidden_states,return_dict)
         
        
        preded = self.downstream_prediction(hidden_states)
        labels['phase'] = status_seq  
        target = {}
        for key, val in labels.items():
            if len(val.shape)>1 and self.min_used_length and val.shape[-1] > self.min_used_length:
                target[key] = val[...,self.min_used_length:]
            elif len(val.shape) == 1:
                target[key] = val.unsqueeze(1) # ==> (B,1) ==> (B,L)
                if preded[key].shape[1] > 1 :
                    target[key]= target[key].repeat(1,preded[key].shape[1])
            else:
                target[key] = val
        loss, error_record, prediction = self.evaluate_error(target, preded,get_prediction=get_prediction)
        #print(loss)
        return SignalOutput(
            loss=loss,
            logits=preded['phase'] if 'phase' in preded else None,
            state=rwkv_outputs.state,
            hidden_states=rwkv_outputs.hidden_states,
            attentions=rwkv_outputs.attentions,
            error_record=error_record,
            prediction=prediction
        )

    # fancy initialization of all lin & emb layer in the module
    # def _init_weights(self, m):
    #     config = self.config
    #     if isinstance(m, nn.Embedding):
    #         shape = m.weight.data.shape
    #         gain = 1.0
    #         scale = 1.0  # extra scale for gain
    #         gain = math.sqrt(max(shape[0], shape[1]))
    #         if shape[0] == config.vocab_size and shape[1] == config.n_embd:  # token emb?
    #             scale = 1e-4
    #         else:
    #             scale = 0
    #     elif isinstance(m, nn.Linear):
    #         shape = m.weight.data.shape
    #         gain = 1.0
    #         scale = 1.0  # extra scale for gain
    #         if m.bias is not None:
    #             m.bias.data.zero_()
    #         if shape[0] > shape[1]:
    #             gain = math.sqrt(shape[0] / shape[1])
    #         if shape[0] == config.vocab_size and shape[1] == config.n_embd:  # final projection?
    #             scale = 0.5
    #         if hasattr(m, 'scale_init'):
    #             scale = m.scale_init
    #         gain *= scale
    #         if scale == -999:
    #             nn.init.eye_(m.weight)
    #         elif gain == 0:
    #             # zero init is great for some RWKV matrices
    #             nn.init.zeros_(m.weight)
    #         elif gain > 0:
    #             nn.init.orthogonal_(m.weight, gain=gain)
    #         else:
    #             nn.init.normal_(m.weight, mean=0.0, std=-scale)
    
    #     elif isinstance(m, nn.Conv2d):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         #torch.nn.init.constant_(m.weight,0.5)
    #         if m.bias is not None:
    #             torch.nn.init.zeros_(m.bias)
    
class RwkvForSignal(RwkvForSignalBase):
    def __init__(self, args, downstream_pool):
        super().__init__(config)
        config = RwkvConfig(vocab_size=args.vocab_size,
                            context_length=args.max_length,
                            hidden_size=args.hidden_size,
                            intermediate_size=args.intermediate_size,
                            attention_hidden_size=args.attention_hidden_size,
                            num_hidden_layers=args.num_hidden_layers, wave_channel=args.wave_channel)
        self.rwkv = RwkvModel(config)
        self.min_used_length = min_used_length = 100
        # 1. the type_embedding part serve a dense tensor initial from norm distribution
        # 2. the wave_embedding part use linear convert from a non-unified data may better get nonlinear activate
        self.wave_embedding = nn.Sequential(nn.Linear(3, config.hidden_size), nn.Tanh())
        self.embedding_merge = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.build_downstring_task(config, downstream_pool)
        self.post_init()


class RwkvforSignal_Sequence(RwkvForSignalBase):
    def __init__(self, config, downstream_pool):
        super().__init__(config)
        self.rwkv = RwkvModel(config)
        self.min_used_length = min_used_length = 0
        # 1. the type_embedding part serve a dense tensor initial from norm distribution
        # 2. the wave_embedding part use linear convert from a non-unified data may better get nonlinear activate
        self.wave_embedding  = nn.Sequential(nn.Linear(3, config.hidden_size), nn.Tanh())
        self.embedding_merge = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.build_downstring_task(config, downstream_pool)
        #self.supports_gradient_checkpointing = False
        self.post_init()

    def kernel_forward(self, inputs_embeds,state,use_cache,output_attentions,output_hidden_states,return_dict):
        rwkv_outputs = self.rwkv(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = rwkv_outputs[0][:,-1:,:] #(B,1,hidden_state)
        return rwkv_outputs, hidden_states
        

class RwkvForPrediction(RwkvForSignalBase):
    def __init__(self, args, downstream_pool):
        config = RwkvConfig(vocab_size=args.vocab_size,
                            context_length=args.max_length,
                            hidden_size=args.hidden_size,
                            intermediate_size=args.intermediate_size,
                            attention_hidden_size=args.attention_hidden_size,
                            num_hidden_layers=args.num_hidden_layers)
        super().__init__(config)
        self.rwkv = RwkvModel(config)
        self.min_used_length = min_used_length = 100
        # 1. the type_embedding part serve a dense tensor initial from norm distribution
        # 2. the wave_embedding part use linear convert from a non-unified data may better get nonlinear activate
        self.wave_embedding  = nn.Sequential(nn.Linear(args.wave_channel, config.hidden_size), nn.Tanh())
        self.rwkv.embeddings = None
        self.embedding_merge = None
        self.build_downstring_task(config, downstream_pool)
        self.post_init()

    def get_composed_input_embedding(self, status_seq, waveform_seq):
        return self.wave_embedding(waveform_seq)


class RwkvOnlyWave(RwkvForPrediction):

    def __init__(self, config, downstream_pool):
        super().__init__(config, downstream_pool)
        self.rwkv.embeddings = None
        self.embedding_merge = None

    def get_composed_input_embedding(self, status_seq, waveform_seq):
        return self.wave_embedding(waveform_seq)
    
    
class RwkvPred_MSF_Phase(RwkvForSignalBase):
    '''
    If use this mode, the sequence lenght should be fixed since we use CNN. 
    The output for the hidden state should be (B, 1, D)
    '''

    def __init__(self, args, downstream_pool):
        config = RwkvConfig(vocab_size=args.vocab_size,
                            context_length= args.max_length//3,#2000,
                            hidden_size=args.hidden_size,
                            intermediate_size=args.intermediate_size,
                            attention_hidden_size=args.attention_hidden_size,
                            num_hidden_layers=args.num_hidden_layers)
        super().__init__(config)
        self.min_used_length = 0
        self.wave_embedding = nn.Sequential(
            MultiScalerFeature(args.wave_channel, args.msf_inner_dim,
                               args.msf_inner_dim, abs_feature=True, stride=3,  # 6000->2000
                               scalers=[3, 5, 7],  # args.msf_levels,
                               cnn_type='symmetry'),
            nn.Tanh(),
            nn.LayerNorm(args.msf_inner_dim),
            MultiScalerFeature(args.msf_inner_dim, args.msf_inner_dim,
                               args.hidden_size, abs_feature=False, stride=1,
                               scalers=[3, 5, 7],  # args.msf_levels,
                               cnn_type='vallina'),
            nn.Tanh(),
            nn.LayerNorm(args.hidden_size),
        )
        
        self.rwkv = RwkvModel(config)
        self.rwkv.embeddings = None
        self.embedding_merge = None
        self.merge_token     = args.merge_token
        # 1. the type_embedding part serve a dense tensor initial from norm distribution
        # 2. the wave_embedding part use linear convert from a non-unified data may better get nonlinear activate
        self.build_downstring_task(config, downstream_pool)
        self.post_init()

    def deal_with_autoregress_sequence(self,status_seq):
        return status_seq
    
    def get_composed_input_embedding(self, status_seq, waveform_seq):
        if len(status_seq.shape) == 2:
            status_seq = status_seq.unsqueeze(-1)
        _input = torch.cat([waveform_seq, status_seq], -1)
        enc_out = self.wave_embedding(_input)
        #enc_out = torch.nn.functional.pad(enc_out, (0, 0, 1000, 0, 0, 0))

        return enc_out
    
    def kernel_forward(self, inputs_embeds, state, use_cache, output_attentions, output_hidden_states, return_dict):
        rwkv_outputs = self.rwkv(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = rwkv_outputs[0]
        if self.merge_token == 'average':
            hidden_states = hidden_states.mean(1, keepdims=True)
        elif self.merge_token == 'last':
            hidden_states = hidden_states[:,-1:,:]
        elif self.merge_token == 'first':
            hidden_states = hidden_states[:,0:1,:]
        else:
            raise ValueError("merge_token only support average, last, first")

        return rwkv_outputs, hidden_states   # -> (B, 1, hidden_size)
