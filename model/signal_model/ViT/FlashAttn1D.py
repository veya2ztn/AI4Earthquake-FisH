# Copyright (c) 2022, Tri Dao.
# Inspired by / adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
import math
import re
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.helpers import named_apply
from torch.nn.init import trunc_normal_
from torchvision.ops import StochasticDepth
from torch import _assert

from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None
from trace_utils import print0
from rotary_embedding_torch import RotaryEmbedding
import os
from flash_attn.layers.patch_embed import PatchEmbed as PatchEmbed2D
def get_local_rank(args=None):
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    return local_rank


class RotatyMHA(MHA):
    """Multi-head self-attention and cross-attention"""

    def __init__(self,*args ,**kargs):
        rotary_emb_dim = kargs.get('rotary_emb_dim', 0)
        assert rotary_emb_dim > 0, "must use rotary embding"
        cross_attn     = kargs.get('cross_attn')
        kargs['rotary_emb_dim'] = 0
        kargs['cross_attn'] = False
        if cross_attn:
            print0("we will force set cross_attn False!")
        super().__init__(*args,**kargs)
        self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim)
        self.rotary_emb_dim = rotary_emb_dim
    
    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
            assert not self.dwconv
            assert self.rotary_emb_dim == 0
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.use_flash_attn
        assert inference_params is None

        kwargs = (
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen, **kwargs}
            if self.use_flash_attn
            else {"key_padding_mask": key_padding_mask, **kwargs}
        )
        seqlen_offset = 0 if inference_params is None else inference_params.sequence_len_offset
        assert not self.cross_attn and self.num_heads_kv == self.num_heads
        assert x_kv is None and mixer_subset is None
        if not self.return_residual:
            qkv = self.Wqkv(x)
        else:
            qkv, x = self.Wqkv(x)
        if self.dwconv:
            qkv = rearrange(self.dwconv_qkv(rearrange(qkv, "b s d -> b d s"))[..., :-2], "b d s -> b s d").contiguous()
        
        assert (inference_params is None or inference_params.sequence_len_offset == 0 or not inference_params.fused_ft_kernel)
        assert self.rotary_emb_dim > 0

        qkv = rearrange(qkv, "B L (three h d) -> three B h L d", three=3, d=self.head_dim)
        q, k, v = qkv
        q = self.rotary_emb.rotate_queries_or_keys(q) #(B h L d)
        k = self.rotary_emb.rotate_queries_or_keys(k)  # (B h L d)
        qkv     = torch.stack([q,k,v],0)
        qkv = rearrange(qkv, "three B h L d -> B L three h d", three=3, d=self.head_dim)
        
        if inference_params is None:
            if not self.checkpointing:
                context = self.inner_attn(qkv, **kwargs)
            else:
                context = torch.utils.checkpoint.checkpoint(
                    self.inner_attn, qkv, **kwargs)
        else:
            q = qkv[:, :, 0]
            kv = self._update_kv_cache(qkv[:, :, 1:], inference_params)
            # If we're processing the prompt, causal=None (use self.causal).
            # If we're decoding, then causal=False.
            causal = None if inference_params.sequence_len_offset == 0 else False
            context = self.inner_cross_attn(q, kv, causal=causal)
            
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)


class PatchEmbed1D(nn.Module):
    """1D Signal to Patch Embedding"""

    def __init__(
        self,
        signal_length=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        fused_bias_fc=False,
    ):
        super().__init__()
        signal_length      = signal_length
        patch_size         = patch_size
        self.signal_length = signal_length
        self.patch_size    = patch_size
        self.grid_size     = signal_length // patch_size
        self.num_patches   = self.grid_size
        self.flatten = flatten
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")

        linear_cls = nn.Linear if not fused_bias_fc or not bias else FusedDense
        self.proj = linear_cls(in_chans * patch_size, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H = x.shape
        # _assert(
        #     H == self.signal_length,
        #     f"Input image height ({H}) doesn't match model ({self.signal_length}).",
        # )

        x = self.proj(rearrange(x,"b c (h p1)  -> b h (c p1)",p1=self.patch_size))
        x = self.norm(x)
        return x

        
# def create_mixer_cls(num_heads, qkv_bias, attn_drop, use_flash_attn, fused_bias_fc, out_proj_bias=True,cross_attn=False, rotary_emb_dim = 0):
#     if rotary_emb_dim == 0:
#         mixer_cls = partial(
#             MHA,
#             num_heads=num_heads,
#             cross_attn=cross_attn,
#             qkv_proj_bias=qkv_bias,
#             out_proj_bias=out_proj_bias,
#             dropout=attn_drop,
#             fused_bias_fc=fused_bias_fc,
#             use_flash_attn=use_flash_attn,
#             rotary_emb_dim=0,#<---- enable if >0
#             rotary_emb_base=10000.0,
#         )   
#     else:#<---- for the latest version of flash attn, it has build in rotaty embding
#         mixer_cls = partial(
#             RotatyMHA,
#             num_heads=num_heads,
#             cross_attn=cross_attn,
#             qkv_proj_bias=qkv_bias,
#             dropout=attn_drop,
#             out_proj_bias=out_proj_bias,
#             fused_bias_fc=fused_bias_fc,
#             use_flash_attn=use_flash_attn,
#             rotary_emb_dim=rotary_emb_dim,#<---- enable if >0
#             rotary_emb_base=10000.0,
#         )   
#     return mixer_cls

def create_mixer_cls(num_heads, qkv_bias, attn_drop, use_flash_attn, fused_bias_fc, out_proj_bias=True,cross_attn=False, rotary_emb_dim = 0):
    mixer_cls = partial(
            MHA,
            num_heads=num_heads,
            cross_attn=cross_attn,
            qkv_proj_bias=qkv_bias,
            out_proj_bias=out_proj_bias,
            dropout=attn_drop,
            fused_bias_fc=fused_bias_fc,
            use_flash_attn=use_flash_attn,
            rotary_emb_dim=rotary_emb_dim,#<---- enable if >0
            rotary_emb_base=10000.0,
        )   
    return mixer_cls

def create_mlp_cls(embed_dim, mlp_ratio, act_layer, fused_mlp, mlp_bias=True,):
    inner_dim = int(embed_dim * mlp_ratio)
    if not fused_mlp:
        mlp_cls = partial(Mlp, hidden_features=inner_dim,bias1=mlp_bias, bias2=mlp_bias,
                          activation=act_layer())
    else:
        mlp_cls = partial(FusedMLP, hidden_features=inner_dim,bias1=mlp_bias, bias2=mlp_bias)
    return mlp_cls


def create_block(
    embed_dim,
    num_heads,
    mlp_ratio,
    qkv_bias,
    drop_rate,
    attn_drop_rate,
    drop_path1,
    drop_path2,
    norm_layer,
    act_layer,
    use_flash_attn,
    fused_bias_fc,
    fused_mlp,
    fused_dropout_add_ln,
    layer_idx=None,
    n_layer=None,
    last_layer_subset=False,
    rotary_emb_dim=0,
    out_proj_bias=True,
    mlp_bias=True,
):
    mixer_cls = create_mixer_cls(
        num_heads,
        qkv_bias,
        attn_drop_rate,
        use_flash_attn,
        fused_bias_fc,
        out_proj_bias=out_proj_bias,
        rotary_emb_dim=rotary_emb_dim,
        cross_attn=(False and last_layer_subset and layer_idx == n_layer - 1),
    )
    mlp_cls = create_mlp_cls(embed_dim, mlp_ratio, act_layer, fused_mlp, mlp_bias=mlp_bias)
    # TD [2022-10-15]: Force residual in fp32 in case of DeepSpeed
    block = Block(
        embed_dim,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_layer,
        prenorm=True,
        resid_dropout1=drop_rate,
        resid_dropout2=drop_rate,
        drop_path1=drop_path1,
        drop_path2=drop_path2,
        fused_dropout_add_ln=fused_dropout_add_ln,
        residual_in_fp32=True,
    )
    return block


class Transformer(nn.Module):
    """
        Input is signal like (B, L, D)
    """

    def __init__(
        self,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=None,
        class_token=True,
        no_embed_class=False,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        norm_layer=None,
        act_layer=None,
        use_flash_attn=False,
        fused_bias_fc=False,
        fused_mlp=False,
        fused_dropout_add_ln=False,
        rotary_emb_dim=0,
        out_proj_bias=True,
        mlp_bias=True,
    ):
        """
        Args:
            signal_length (int, tuple): input signal size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool == "token", "Only support pooling with CLS token"
        #assert class_token
        assert init_values is None, "LayerScale is not supported yet"
        assert weight_init == ""
        assert fc_norm is None
        # pre_norm seems redundant, as there's a LayerNorm right at the start of each block, idk
        assert not pre_norm
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        
        self.global_pool       = global_pool
        self.num_features      = (self.embed_dim) = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class    = no_embed_class

        

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        #dpr = [0]*depth #<------disable dropout

        
        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.patch_embed = None
        self.blocks = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    drop_path1=dpr[i - 1] if i > 0 else 0.0,
                    drop_path2=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_flash_attn=use_flash_attn,
                    fused_bias_fc=fused_bias_fc,
                    fused_mlp=fused_mlp,
                    fused_dropout_add_ln=fused_dropout_add_ln,
                    layer_idx=i,
                    n_layer=depth,
                    last_layer_subset=(global_pool == "token"),
                    rotary_emb_dim=rotary_emb_dim,
                    out_proj_bias=out_proj_bias,
                    mlp_bias=mlp_bias,
                )
                for i in range(depth)
            ]
        )

        self.dropout   = nn.Dropout(p=drop_rate)
        self.drop_path = StochasticDepth(p=dpr[-1], mode="row")
        self.norm = norm_layer(embed_dim)

        self.fused_dropout_add_ln = fused_dropout_add_ln
        if self.fused_dropout_add_ln and dropout_add_layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")

        

        

    def init_weights(self, mode=""):
        assert mode == ""
        if self.pos_embed:trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _pos_embed(self, x):
        raise NotImplementedError

    def forward_features(self, x, all_tokens=True):
        """
        If all_tokens==False and self.global_pool == 'token', we only return the features for the
        cls token.
        """
        x = self.patch_embed(x)
        hidden_states = self._pos_embed(x)
        residual = None
        if self.global_pool != "token" or all_tokens:
            # if True:
            for block in self.blocks:
                hidden_states, residual = block(hidden_states, residual)
        else:
            for block in self.blocks[:-1]:
                hidden_states, residual = block(hidden_states, residual)
            # For the last layer, we only want the 1st token of the output. So we do cross-attention
            # where the query is the oken and1st t the key/value is the whole sequence.
            hidden_states, residual = self.blocks[-1](
                hidden_states, residual, mixer_subset=slice(0, 1)
            )
        if not self.fused_dropout_add_ln:
            residual = self.drop_path(self.dropout(hidden_states)) + residual
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            if self.drop_path.p == 0 or not self.training:
                rowscale = None
            else:
                rowscale = self.drop_path(
                    torch.ones(
                        hidden_states.shape[:-1],
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                )
            # Set prenorm=False here since we don't need to the residual
            hidden_states = dropout_add_layer_norm(
                hidden_states,
                residual,
                self.norm.weight,
                self.norm.bias,
                self.dropout.p if self.training else 0.0,
                self.norm.eps,
                rowscale=rowscale,
                prenorm=False,
                residual_in_fp32=True,
            )
        return hidden_states

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x, all_tokens=False)
        x = self.forward_head(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        patch_embed_weight = state_dict["patch_embed.proj.weight"]
        if patch_embed_weight.dim() == 4:
            # convert from Conv2d to Linear
            state_dict["patch_embed.proj.weight"] = rearrange(
                patch_embed_weight, "o c h w -> o (c h w)"
            )

        def key_mapping_attn(key):
            key = re.sub(r"^blocks.(\d+).attn.qkv.",
                         r"blocks.\1.mixer.Wqkv.", key)
            key = re.sub(r"^blocks.(\d+).attn.proj.",
                         r"blocks.\1.mixer.out_proj.", key)
            return key

        state_dict = OrderedDict((key_mapping_attn(k), v)
                                 for k, v in state_dict.items())
        n_layer = len(self.blocks)
        # Convert from Wqkv to Wq and Wkv for cross attention (last layer)
        if (
            self.blocks[-1].mixer.cross_attn
            and f"blocks.{n_layer - 1}.mixer.Wqkv.weight" in state_dict
        ):
            Wqkv = state_dict.pop(f"blocks.{n_layer - 1}.mixer.Wqkv.weight")
            bqkv = state_dict.pop(f"blocks.{n_layer - 1}.mixer.Wqkv.bias")
            state_dict[f"blocks.{n_layer - 1}.mixer.Wq.weight"] = Wqkv[: self.embed_dim]
            state_dict[f"blocks.{n_layer - 1}.mixer.Wkv.weight"] = Wqkv[self.embed_dim:]
            state_dict[f"blocks.{n_layer - 1}.mixer.Wq.bias"] = bqkv[: self.embed_dim]
            state_dict[f"blocks.{n_layer - 1}.mixer.Wkv.bias"] = bqkv[self.embed_dim:]
        return super().load_state_dict(state_dict, strict=strict)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()

class SignalTransformer(Transformer):
    def __init__(self,signal_length=6000,patch_size=16,in_chans=3,
                 num_classes=1000, embed_layer=PatchEmbed1D, global_pool="token",
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 init_values=None,
                 class_token=True,
                 no_embed_class=False,
                 pre_norm=False,
                 fc_norm=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 weight_init="",
                 norm_layer=None,
                 act_layer=None,
                 use_flash_attn=False,
                 fused_bias_fc=False,
                 fused_mlp=False,
                 fused_dropout_add_ln=False,
                 rotary_emb_dim=0,out_proj_bias=True,
                 disable_bias=False,mlp_bias=True):
        if disable_bias:
            print0(f"we disable all bias")
            qkv_bias = out_proj_bias= mlp_bias = False
            norm_layer = partial(UnitLayerNorm, eps=1e-6) if disable_bias else norm_layer
        super().__init__(global_pool          =  global_pool         ,
                         embed_dim            =  embed_dim           ,
                         depth                =  depth              ,
                         num_heads            =  num_heads           ,
                         mlp_ratio            =  mlp_ratio           ,
                         qkv_bias             =  qkv_bias            ,
                         init_values          =  init_values         ,
                         class_token          =  class_token         ,
                         no_embed_class       =  no_embed_class      ,
                         pre_norm             =  pre_norm            ,
                         fc_norm              =  fc_norm             ,
                         drop_rate            =  drop_rate           ,
                         attn_drop_rate       =  attn_drop_rate      ,
                         drop_path_rate       =  drop_path_rate      ,
                         weight_init          =  weight_init         ,
                         norm_layer           =  norm_layer          ,
                         act_layer            =  act_layer           ,
                         use_flash_attn       =  use_flash_attn      ,
                         fused_bias_fc        =  fused_bias_fc       ,
                         fused_mlp            =  fused_mlp           ,
                         fused_dropout_add_ln =  fused_dropout_add_ln,
                         rotary_emb_dim       = rotary_emb_dim,
                         out_proj_bias=out_proj_bias,mlp_bias=mlp_bias)
        self.num_classes = num_classes
        patch_embed_extra_kwargs = (
                {"fused_bias_fc": fused_bias_fc} if embed_layer is PatchEmbed1D else {}
            )
            

        self.patch_embed = embed_layer(
            signal_length=signal_length,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias= (not pre_norm) and (not disable_bias),  # disable bias if pre-norm is used (e.g. CLIP)
            **patch_embed_extra_kwargs,
        )
        num_patches = self.patch_embed.num_patches
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.embed_len = embed_len
        self.pos_embed = None
        self.head      = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights(weight_init)
        
    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return x
        

class Signal2DTransformer(Transformer):
    def __init__(self,img_size=256,patch_size=16,in_chans=3,
                 num_classes=1000, embed_layer=PatchEmbed2D, global_pool="token",
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 init_values=None,
                 class_token=True,
                 no_embed_class=False,
                 pre_norm=False,
                 fc_norm=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 weight_init="",
                 norm_layer=None,
                 act_layer=None,
                 use_flash_attn=False,
                 fused_bias_fc=False,
                 fused_mlp=False,
                 fused_dropout_add_ln=False,
                 rotary_emb_dim=0,out_proj_bias=True,
                 disable_bias=False,mlp_bias=True):
        if disable_bias:
            print0(f"we disable all bias")
            qkv_bias = out_proj_bias= mlp_bias = False
            norm_layer = partial(UnitLayerNorm, eps=1e-6) if disable_bias else norm_layer
        super().__init__(global_pool          =  global_pool         ,
                         embed_dim            =  embed_dim           ,
                         depth                =  depth              ,
                         num_heads            =  num_heads           ,
                         mlp_ratio            =  mlp_ratio           ,
                         qkv_bias             =  qkv_bias            ,
                         init_values          =  init_values         ,
                         class_token          =  class_token         ,
                         no_embed_class       =  no_embed_class      ,
                         pre_norm             =  pre_norm            ,
                         fc_norm              =  fc_norm             ,
                         drop_rate            =  drop_rate           ,
                         attn_drop_rate       =  attn_drop_rate      ,
                         drop_path_rate       =  drop_path_rate      ,
                         weight_init          =  weight_init         ,
                         norm_layer           =  norm_layer          ,
                         act_layer            =  act_layer           ,
                         use_flash_attn       =  use_flash_attn      ,
                         fused_bias_fc        =  fused_bias_fc       ,
                         fused_mlp            =  fused_mlp           ,
                         fused_dropout_add_ln =  fused_dropout_add_ln,
                         rotary_emb_dim       = rotary_emb_dim,
                         out_proj_bias=out_proj_bias,mlp_bias=mlp_bias)
        self.num_classes = num_classes
        patch_embed_extra_kwargs = (
                {"fused_bias_fc": fused_bias_fc} if embed_layer is PatchEmbed2D else {}
            )

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias= (not pre_norm) and (not disable_bias),  # disable bias if pre-norm is used (e.g. CLIP)
            **patch_embed_extra_kwargs,
        )
        num_patches = self.patch_embed.num_patches
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.embed_len = embed_len
        self.pos_embed = None
        self.head      = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights(weight_init)
        
    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return x

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
