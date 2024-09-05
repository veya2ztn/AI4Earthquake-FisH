import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from timm.models.layers import drop_path
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import top_k_top_p_filtering

from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from torch.nn import LayerNorm

from .configuration_retnet import RetNetConfig

logger = logging.get_logger(__name__)


# helper functions
def split_heads(tensors, bsz, seqlen, num_heads):
    assert isinstance(tensors, (tuple, list))
    return [x.view(bsz, seqlen, num_heads, -1).transpose(1, 2) for x in tensors]


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "swish":
        return F.silu
    else:
        raise NotImplementedError


from .self_retention import SelfRetentionV2 as SelfRetention
from .self_retention import RetNetRelPosV2 as RetNetRelPos
from .self_retention import RMSNorm, groupnorm_pool

class MultiScaleRetention(nn.Module):

    def __init__(
        self,
        config: RetNetConfig,
        gate_fn="swish",
        use_bias=False,
        tensor_parallel=False,
        
    ):
        super().__init__()
        self.config = config
        
        self.embed_dim = config.decoder_embed_dim
        self.value_dim = config.decoder_value_embed_dim
        self.num_heads = config.decoder_retention_heads
        self.head_dim = self.value_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim**-0.5

        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=use_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.value_dim, bias=use_bias)
        self.g_proj = nn.Linear(self.embed_dim, self.value_dim, bias=use_bias)

        self.out_proj = nn.Linear(self.value_dim, self.embed_dim, bias=use_bias)
        self.self_retention= SelfRetention(config)
        self.reset_parameters()

        assert not tensor_parallel
        #self.decay_proj = nn.Linear(self.num_heads, self.num_heads, bias=False) if tensor_parallel else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)

    

    def forward(
        self,
        hidden_states: torch.Tensor,
        rel_pos: Tuple[Tuple[torch.Tensor]],
        retention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        forward_impl: str = 'parallel',
        monitor_retention: Optional[bool] = False,
        output_increment: Optional[bool] = False,
        output_k_score_in_column_space: Optional[str] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        B, T, H = hidden_states.size()
        
        (sin, cos), decay_mask = rel_pos
        
        # projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)
        
        score_activate_k = None
        if output_k_score_in_column_space:
            A = self.v_proj.weight
            assert A.shape[1] > A.shape[0], "you must use correct config to ensure v_proj.weight is tall matrix, that map Large Input dim to Small dim"
            assert self.v_proj.bias is None, "please do not use bias in v_proj, otherwise the result will be wrong"
            Q,R = torch.linalg.qr(A)
            # if 'normlize' in output_k_score_in_column_space:
            #     hidden_states = hidden_states/(hidden_states.norm(dim=-1,keepdim=True)+1e-6)
            if 'detach' in output_k_score_in_column_space:
                hidden_states = hidden_states.detach()
            score_activate_k = ((hidden_states[...,:A.shape[0]]@Q)@Q.T).norm(dim=-1) #(B,L)
            #retention layer will normlize each hidden_state(B,L,D) vector (1,D) to norm = np.sqrt(D)
            score_activate_k = score_activate_k/(np.sqrt(hidden_states.shape[-1]))

        # multi-head
        q, k, v = split_heads((q, k, v), B, T, self.num_heads)


        k = k*self.scaling  # for scaled dot product
        # rotate
        # NOTE: theta_shift has bug with mps device.
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        retention_out, retention_monitor, curr_kv, increment  = self.self_retention(qr, kr, v, decay_mask,
                        past_key_value=past_key_value, 
                        retention_mask=retention_mask,
                        forward_impl = forward_impl,
                        output_increment=output_increment,
                        monitor_retention=monitor_retention)
         
        # concaat heads
        # normed = self.group_norm(retention_out).reshape(B, T, self.value_dim) 
        # ## <--- it is better move the groupnorm into the function, thus the result obtain from different method will be same.
        # ##      otherwise, only the recurrent and parallel is same, but chunkwise is wrong.
        # out gate & proj
        out = self.gate_fn(g) * retention_out.reshape(B, T, self.value_dim)
        out = self.out_proj(out)

        outputs = (out, curr_kv, retention_monitor, increment, score_activate_k)

        return outputs
    
import torch.nn.functional as F
import numbers
class UnitLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        #self.register_buffer('weight', weight)
        
        bias   = torch.zeros(self.normalized_shape, **factory_kwargs)
        self.register_buffer('bias', bias)
    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps},'.format(**self.__dict__)

class FeedForwardNetwork(nn.Module):

    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
        layernorm_eps,
        subln=False,
        use_rms_norm=False,
        use_bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim, bias=use_bias)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, bias=use_bias)
        layernormModule=LayerNorm if use_bias else UnitLayerNorm 
        self.ffn_layernorm = layernormModule(ffn_dim, eps=layernorm_eps) if subln else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x)
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x

class GLU(nn.Module):

    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.gate = nn.Linear(self.embed_dim, ffn_dim, bias=False)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.gate.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        g = self.gate(x)
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x) * g
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return "p={}".format(self.drop_prob)




class RetNetDecoderLayer(nn.Module):

    def __init__(self, config: RetNetConfig, depth: int, tensor_parallel: bool = False):
        super().__init__()
        self.config = config
        self.embed_dim = config.decoder_embed_dim
        self.dropout_module = torch.nn.Dropout(config.dropout)
        self.drop_path = DropPath(np.linspace(0, config.drop_path_rate, config.decoder_layers)[depth]) if config.drop_path_rate > 0 else None

        self.retention = MultiScaleRetention(config,use_bias=False,tensor_parallel=tensor_parallel)

        self.normalize_before = config.decoder_normalize_before
        
        self.retention_layer_norm = groupnorm_pool[config.groupnorm_type](self.embed_dim, eps=config.layernorm_eps)

        self.ffn_dim = config.decoder_ffn_embed_dim

        self.ffn = self.build_ffn()

        self.final_layer_norm = groupnorm_pool[config.groupnorm_type](self.embed_dim, eps=config.layernorm_eps)

        self.alpha = math.pow(2.0 * config.decoder_layers, 0.25) if config.deepnorm else 1.0

    def build_ffn(self):
        if self.config.use_glu:
            return GLU(
                self.embed_dim,
                self.ffn_dim,
                self.config.activation_fn,
                self.config.dropout,
                self.config.activation_dropout,
            )
        else:
            return FeedForwardNetwork(
                self.embed_dim,
                self.ffn_dim,
                self.config.activation_fn,
                self.config.dropout,
                self.config.activation_dropout,
                self.config.layernorm_eps,
                self.config.subln,
                self.config.use_ffn_rms_norm,
                not self.config.disable_all_bias,
            )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        hidden_states: torch.Tensor,
        retention_rel_pos: Tuple[Tuple[torch.Tensor]],
        retention_mask: Optional[torch.Tensor] = None,
        forward_impl: str = 'parallel',
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        monitor_retention: Optional[bool] = False,
        output_increment: Optional[bool] = False,
        output_k_score_in_column_space: Optional[str] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        residual = hidden_states
        if output_k_score_in_column_space:
            assert self.normalize_before, "output_k_score_in_column_space only support normalize_before=True"
        if self.normalize_before:
            hidden_states = self.retention_layer_norm(hidden_states)

        msr_outs = self.retention(hidden_states,
                                  retention_rel_pos,
                                  retention_mask=retention_mask,
                                  past_key_value=past_key_value,
                                  forward_impl=forward_impl,
                                  monitor_retention=monitor_retention,
                                  output_increment=output_increment,
                                  output_k_score_in_column_space=output_k_score_in_column_space,)
        hidden_states,curr_kv,retention_weights, increment,score_activate_k = msr_outs

        hidden_states = self.dropout_module(hidden_states)

        if self.drop_path is not None:
            hidden_states = self.drop_path(hidden_states)

        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.retention_layer_norm(hidden_states)

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.ffn(hidden_states)

        if self.drop_path is not None:
            hidden_states = self.drop_path(hidden_states)

        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, curr_kv, retention_weights, increment,score_activate_k)

        return outputs


class RetNetPreTrainedModel(PreTrainedModel):
    # copied from LlamaPretrainedModel
    config_class = RetNetConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RetNetDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        """
        Following original retnet, weights are already initialized in their own
        ways within their own init.
        """
        # pass
        # below is copied from LlamaPretrainedModel
        if self.config.decoder_layers>4:return
        std = 0.02#self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RetNetModel):
            module.gradient_checkpointing = value


@dataclass
class RetNetOutputWithPast(ModelOutput):
    """
    class for RetNet model's outputs that may also contain a past key/values (to speed up sequential decoding).

    config:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, decoder_embed_dim)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            decoder_embed_dim)` is output.
        past_key_values (`List(Dict(str, torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            - "prev_key_value": shape=(bsz * num_head * v_dim * qk_dim)
            - "scale": shape=((1 or bsz) * num_head * 1 * 1)

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks)
            that can be used (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, decoder_embed_dim)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        retentions (`tuple(torch.FloatTensor)`, *optional*, returned when `monitor_retention=True` is passed or when `config.monitor_retention=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Retentions weights, used for visualization.

        attentions (`tuple(torch.FloatTensor)`, *optional*, for backward compatibility. Same as retentions.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[List[Dict[str, torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    retentions: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    increment: Optional[Tuple[torch.FloatTensor]] = None
    score_activate_k: Optional[Tuple[torch.FloatTensor]] = None

class RetNetModel(RetNetPreTrainedModel):

    def __init__(self,
                 config: RetNetConfig,
                 embed_tokens: nn.Embedding = None,
                 tensor_parallel: bool = False):
        super().__init__(config)
        self.config = config

        self.dropout_module = torch.nn.Dropout(config.dropout)

        self.embed_dim = config.decoder_embed_dim
        self.embed_scale = 1.0 if config.no_scale_embedding else math.sqrt(self.embed_dim)

        if embed_tokens is None:
            embed_tokens = nn.Embedding(config.vocab_size, config.decoder_embed_dim,
                                        config.pad_token_id)
        self.embed_tokens = embed_tokens

        if config.layernorm_embedding:
            self.layernorm_embedding = groupnorm_pool[config.groupnorm_type](self.embed_dim, eps=config.layernorm_eps)
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        for i in range(config.decoder_layers):
            self.layers.append(RetNetDecoderLayer(config, depth=i, tensor_parallel=tensor_parallel))

        self.decoder_layers = len(self.layers)

        if config.normalize_at_end:
            self.layer_norm = groupnorm_pool[config.groupnorm_type](self.embed_dim, eps=config.layernorm_eps)
        else:
            self.layer_norm = None

        self.retnet_rel_pos = RetNetRelPos(config)
        self.recurrent_chunk_size = config.recurrent_chunk_size

        if config.deepnorm:
            init_scale = math.pow(8.0 * config.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if ("fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name):
                    p.data.div_(init_scale)

        if config.subln and not config.use_glu:
            init_scale = math.sqrt(math.log(config.decoder_layers * 2))
            for name, p in self.named_parameters():
                if ("fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name):
                    p.data.mul_(init_scale)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward_embedding(
        self,
        input_ids,
        forward_impl,
        inputs_embeds=None,
        past_key_values=None,
    ):
        # if past_key_values is not None:
        if forward_impl == 'recurrent':
            input_ids = input_ids[:, -1:]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed = self.embed_scale * inputs_embeds

        if self.layernorm_embedding is not None:
            embed = self.layernorm_embedding(embed)

        embed = self.dropout_module(embed)

        return embed

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        retention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Dict[str, torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        monitor_retention: Optional[bool] = None,
        output_increment: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_impl: Optional[str] = 'parallel',
        recurrent_chunk_size: Optional[int] = None,
        retention_rel_pos: Optional[Tuple[torch.Tensor]] = None,
        fixed_seq_len: Optional[int] = None,
        output_k_score_in_column_space: Optional[str] = None,
    ) -> Union[Tuple, RetNetOutputWithPast]:

        if monitor_retention is None and output_attentions is not None:
            monitor_retention = output_attentions
        monitor_retention = monitor_retention if monitor_retention is not None else self.config.monitor_retention

        output_hidden_states = (output_hidden_states if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # embed tokens
        if inputs_embeds is None:
            inputs_embeds = self.forward_embedding(input_ids, forward_impl, inputs_embeds,past_key_values)
        else:
            if forward_impl == 'recurrent':
                inputs_embeds = inputs_embeds[:, -1:]
        if retention_mask is None and attention_mask is not None:
            retention_mask = attention_mask
        if retention_mask is not None and forward_impl == 'recurrent':
            retention_mask = retention_mask[:, -1:]

        hidden_states = inputs_embeds

        # handling chunking here
        if recurrent_chunk_size is None:
            recurrent_chunk_size = hidden_states.shape[1]
            #recurrent_chunk_size = self.recurrent_chunk_size
        need_pad_for_chunkwise = (forward_impl == 'chunkwise' and
                                  seq_length % recurrent_chunk_size != 0)
        if need_pad_for_chunkwise:
            padding_len = recurrent_chunk_size - seq_length % recurrent_chunk_size
            slen = seq_length + padding_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding_len))
        else:
            slen = seq_length
        if fixed_seq_len:slen=fixed_seq_len
        # relative position
        if retention_rel_pos is None:
            retention_rel_pos = self.retnet_rel_pos(slen,
                                                    forward_impl=forward_impl,
                                                    recurrent_chunk_size=recurrent_chunk_size,
                                                    retention_mask=retention_mask,
                                                    get_decay_scale=True#not self.training
                                                    )

        # start running through the decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if monitor_retention else None
        # layers * [bsz, num_head, qk_dim, decoder_embed_dim]
        next_decoder_cache = () if use_cache else None
        all_increment  = () if output_increment else None
        all_score_activate_k = () if output_k_score_in_column_space else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError
            else:
                layer_outputs = layer(hidden_states,
                                      retention_rel_pos,
                                      retention_mask=retention_mask,
                                      forward_impl=forward_impl,
                                      past_key_value=past_key_value,
                                      monitor_retention=monitor_retention,
                                      output_increment=output_increment,
                                      output_k_score_in_column_space=output_k_score_in_column_space,)

            hidden_states = layer_outputs[0]

            if use_cache:next_decoder_cache += (layer_outputs[1],)
            if monitor_retention:all_retentions += (layer_outputs[2],)
            if output_increment:all_increment += (layer_outputs[3],)
            if output_k_score_in_column_space:all_score_activate_k  += (layer_outputs[4],)

        next_cache = next_decoder_cache if use_cache else None

        if need_pad_for_chunkwise:
            hidden_states = hidden_states[:, :seq_length, :]

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_retentions, all_increment, all_score_activate_k] if v is not None)
        
        if all_retentions:
            if isinstance(all_retentions[0], dict):
                keys = all_retentions[0].keys()
                all_retentions = {k: torch.stack([ret[k] for ret in all_retentions], dim=-1) for k in keys}
        return RetNetOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            retentions=all_retentions,
            increment=all_increment,
            score_activate_k=all_score_activate_k,
        )
