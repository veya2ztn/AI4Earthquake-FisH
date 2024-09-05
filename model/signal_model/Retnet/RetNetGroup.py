from .RetNet import *


try:
    from flash_attn import __version__ as flash_attn_version
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import (
        flash_attn_func,
        flash_attn_varlen_kvpacked_func,
    )
    from einops import rearrange, repeat
    def scaled_dot_product_attention_flashV2(q, k, v, bsz, q_len, num_heads, head_dim, attention_mask=None):
        #print("alibi_slope ==>", alibi)
        if attention_mask is None:
            output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=False).reshape(
                bsz, q_len, -1
            )
        else:
            #print(attention_mask)
            #q_attention_mask = torch.fill(attention_mask[:, -q_len:],True)
            q_attention_mask = attention_mask[:, -q_len:] ### <--- we are not casual or encode mode, just transformer
            q, indices, cu_q_lens, max_s = unpad_input(q, q_attention_mask) 
            # We can skip concat and call unpad twice but seems better to call unpad only once.
            kv, _, cu_k_lens, max_k = unpad_input(torch.stack((k, v), dim=2), attention_mask)
            output_unpad = flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_q_lens,
                cu_k_lens,
                max_s,
                max_k,
                0.0,
                softmax_scale=None,
                causal=False,
            )
            output_unpad = output_unpad.reshape(-1, num_heads * head_dim)
            output = pad_input(output_unpad, indices, bsz, q_len)
        return output
    
    """It equal to the use of scaled_dot_product_attention
        --------------------------------------------------------
        B = 3
        G = 7
        D = 36
        H = 9
        d = D//H
        q = torch.randn(B, G, D).cuda()
        k = torch.randn(B, G, D).cuda()
        v = torch.randn(B, G, D).cuda()
        station_mask       = torch.ones(B, G)
        station_mask[0,3:] = 0
        station_mask[1,4:] = 0
        station_mask[2,5:] = 0

        import math

        B, G, D = q.shape        
        qi = q.view(B, G, H, d).bfloat16()
        ki = k.view(B, G, H, d).bfloat16()
        vi = v.view(B, G, H, d).bfloat16()
        attention_mask= (station_mask>0).cuda()
        print(attention_mask)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                o = scaled_dot_product_attention_flashV2(qi, ki, vi,  B, G, H, d,  attention_mask= attention_mask)
        print(o[0,2])

        qi = q.view(B, G, H, d).permute(0, 2, 1, 3).cuda()
        ki = k.view(B, G, H, d).permute(0, 2, 1, 3).cuda()
        vi = v.view(B, G, H, d).permute(0, 2, 1, 3).cuda()
        B, H, G, D = qi.shape
        attention_mask = station_mask.cuda()
        attention_mask = get_extended_attention_mask(attention_mask)
        with torch.backends.cuda.sdp_kernel():
            o2 = scaled_dot_product_attention(qi, ki, vi, attention_mask)
        o2 = o2.permute(0, 2, 1, 3).flatten(-2,-1) # (B, G, H, D) ->  (B, G, H*D)
        print(o2[0,2])

        query = qi
        key   = ki
        value = vi
        #attn_mask=get_extended_attention_mask(station_mask).cuda()
        attn_mask=get_extended_attention_mask(torch.einsum("bi,bj->bij",station_mask,station_mask)).cuda()
        dropout_p=0.0
        is_causal=False
        scale=None
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_bias+attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        out =  attn_weight @ value
        out = out.permute(0,2,1,3).flatten(-2,-1)
        print(out[0,2])
    """
    
    flash_attn_is_available = True
except:
    flash_attn_is_available = False
torch.backends.cuda.enable_flash_sdp(True)
from einops import rearrange, repeat
from torch.nn.functional import scaled_dot_product_attention
from .retnet.modeling_retnet import FeedForwardNetwork,RetNetOutputWithPast

class _AttentionLayer(nn.Module):
    def __init__(self, config:SignalModelConfig, first_layer=True):
        super().__init__()
        self.config = config
        self.num_attention_heads = self.config.Backbone.num_heads
        self.attention_head_size = int(self.config.cross_hidden_size / self.num_attention_heads)
        self.build_distance_qkv(config)
        self.build_hidden_state_qkv(config,first_layer=first_layer)
        self.build_layer_norm(config)
        self.build_proj_out(config)


    def attention(self, q, k, v, attention_mask,use_flashattn=True):
        """
        Please remember only when you apply mask both at 1. attention 2. output, the result is consistancy
        --------------------------------------------
        import torch
        import torch.nn as nn
        from torch.nn.functional import scaled_dot_product_attention

        proj_out = nn.Linear(128, 128)
        layer_norm = nn.LayerNorm(128, eps=1e-16)

        Q = torch.randn(3,8, 128)
        K = torch.randn(3,8, 128)
        V = torch.randn(3,8, 128)
        vector_state = torch.randn(3, 8 ,8,2)
        num_attention_heads = 4
        attention_head_size = 128//num_attention_heads
        attention_mask = torch.ones(3, 8)
        attention_mask[0,3:] = 0
        attention_mask[2,4:] = 0

        def compute_way1(Q,K,V,vector_state,attention_mask):
            B, G, D = Q.shape        
            q = Q.view(B, G, num_attention_heads, attention_head_size).permute(0, 2, 1, 3)
            k = K.view(B, G, num_attention_heads, attention_head_size).permute(0, 2, 1, 3)
            v = V.view(B, G, num_attention_heads, attention_head_size).permute(0, 2, 1, 3)
            #new_bias = attention_mask + bias

            B, H, G, D = q.shape
            assert attention_mask.shape == (B,G), f" now is {attention_mask.shape}"
            attn = scaled_dot_product_attention(q, k, v, get_extended_attention_mask(attention_mask, dtype=q.dtype))
            o = attn.permute(0, 2, 1, 3).flatten(-2,-1) # (B, G, H, D) ->  (B, G, H*D)
            o = layer_norm(o)
            o = proj_out(o) # (B, G, 256)
            mask = attention_mask.to(o)
            mask = (mask[:,:,None]@mask[:,None])
            o = (o@o.mT)
            o = torch.einsum('BGj,Bjk,BGks->BGs',o, mask, vector_state) # (B, G, 256) @ (B, 256, G) @ (B, G, G) @(B, G, 2) -> (B, G, 2)
            o = o[attention_mask>0]
            return attn, o

        def compute_way2(Q,K,V,vector_state,attention_mask):
            B, G, D = Q.shape        
            q = Q.view(B, G, num_attention_heads, attention_head_size).permute(0, 2, 1, 3)
            k = K.view(B, G, num_attention_heads, attention_head_size).permute(0, 2, 1, 3)
            v = V.view(B, G, num_attention_heads, attention_head_size).permute(0, 2, 1, 3)
            #new_bias = attention_mask + bias

            B, H, G, D = q.shape
            assert attention_mask.shape == (B,G), f" now is {attention_mask.shape}"
            mask = attention_mask.to(q)
            mask = (mask[:,:,None]@mask[:,None])
            attn = scaled_dot_product_attention(q, k, v, get_extended_attention_mask(mask, dtype=q.dtype))
            o = attn.permute(0, 2, 1, 3).flatten(-2,-1) # (B, G, H, D) ->  (B, G, H*D)
            o = layer_norm(o)
            o = proj_out(o) # (B, G, 256)
            o = (o@o.mT)
            o = torch.einsum('BGj,Bjk,BGks->BGs',o, mask, vector_state) # (B, G, 256) @ (B, 256, G) @ (B, G, G) @(B, G, 2) -> (B, G, 2)
            o = o[attention_mask>0]
            return attn, o

        attn2,o2=compute_way2(Q,K,V,vector_state,attention_mask)
        attn1,o1=compute_way1(Q,K,V,vector_state,attention_mask)

        print(o2)
        print(o1)
        """
        B, G, D = q.shape        
        # (B, G, G) x (G, D)x(D,G)  x 
        # vector_state -->  (B, G, 2) 
        if flash_attn_is_available and use_flashattn:
            q = q.view(B, G, self.num_attention_heads, self.attention_head_size)
            k = k.view(B, G,   self.num_attention_heads, self.attention_head_size)
            v = v.view(B, G, self.num_attention_heads, self.attention_head_size)
            return  scaled_dot_product_attention_flashV2(q, k, v,  B, G, self.num_attention_heads, self.attention_head_size, 
                                                            attention_mask= attention_mask #(B, G) bool
                                                            )
        else:
            q = q.view(B, G, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            k = k.view(B, G, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            v = v.view(B, G, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            #new_bias = attention_mask + bias
            
            B, H, G, D = q.shape
            assert attention_mask.shape == (B,G)
            attn = scaled_dot_product_attention(q, k, v, self.get_extended_attention_mask(attention_mask, dtype=q.dtype))
            attn = attn.permute(0, 2, 1, 3).flatten(-2,-1) # (B, G, H, D) ->  (B, G, H*D)
            attn[~attention_mask]=0
            return attn 
    
    def get_extended_attention_mask(self, attention_mask, dtype=None):
        if dtype is None:
            dtype = self.dtype

        if attention_mask.dim() == 4:
            extended_attention_mask = attention_mask
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :] # same as (B,S,L,L)            
        else:
            raise ValueError(
                f"Wrong shape attention_mask (shape {attention_mask.shape})"
            )
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask    

    
    def build_distance_qkv(self, config):
        raise NotImplementedError(f"build_distance_qkv should be implemented in subclass")
    
    def build_hidden_state_qkv(self,config,first_layer=True):
        raise NotImplementedError(f"build_distance_qkv should be implemented in subclass")

    def build_layer_norm(self, config):
        raise NotImplementedError(f"build_distance_qkv should be implemented in subclass")

    def build_proj_out(self, config):
        raise NotImplementedError(f"build_distance_qkv should be implemented in subclass")


class SimpleAttention(_AttentionLayer):
    def __init__(self, config:SignalModelConfig):
        super().__init__(config)
        self.query      = nn.Linear(config.cross_hidden_size, config.cross_hidden_size, bias=False)
        self.keyes      = nn.Linear(config.cross_hidden_size, config.cross_hidden_size, bias=False)
        self.value      = nn.Linear(config.cross_hidden_size, config.cross_hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm( config.cross_hidden_size, eps=1e-6)
        self.proj_out   = nn.Linear( config.cross_hidden_size, config.cross_hidden_size, bias=False)

    def forward(self, hidden_state, station_mask, use_flashattn=True):
        B, G, D = hidden_state.shape
        assert station_mask.shape == (B,G), f"please  use bool mask with shape ({B},{G})"
        assert station_mask.dtype == torch.bool, f"please  use bool mask with shape ({B},{G})"
        attention_mask = station_mask#[:,0]
        q = self.query(hidden_state)# (B, G, 42) -> (B, G, 256)
        k = self.keyes(hidden_state)# (B, G, 42) -> (B, G, 256)
        v = self.value(hidden_state)# (B, G, 42) -> (B, G, 256)
        o = self.attention(q, k, v, attention_mask,use_flashattn=use_flashattn) # (B, G, 256)
        o = self.layer_norm(o)
        o = self.proj_out(o) # (B, G, 256)
        return o


class DistanceAttention_Alpha(_AttentionLayer):
    
    def build_distance_qkv(self, config):
        self.query_distance = nn.Linear( config.distance_system_dim, config.cross_hidden_size, bias=False)
        self.keyes_distance = nn.Linear( config.distance_system_dim, config.cross_hidden_size, bias=False)
        self.value_distance = nn.Linear( config.distance_system_dim, config.cross_hidden_size, bias=False)
    
    def build_hidden_state_qkv(self,config,first_layer=True):
        hidden_state_dim = config.Backbone.hidden_size if first_layer else config.cross_hidden_size
        self.query_hidden_state = nn.Linear( hidden_state_dim, config.cross_hidden_size, bias=False)
        self.keyes_hidden_state = nn.Linear( hidden_state_dim, config.cross_hidden_size, bias=False)
        self.value_hidden_state = nn.Linear( hidden_state_dim, config.cross_hidden_size, bias=False)

    def build_layer_norm(self, config):
        self.layer_norm = nn.LayerNorm( config.cross_hidden_size, eps=1e-6)

    def build_proj_out(self, config):
        # ffm_config = {
        #     'embed_dim':config.cross_hidden_size,
        #     'ffn_dim':config.cross_hidden_size,
        #     'activation_fn':'swish', #<<<--- notice here
        #     'dropout':0,
        #     'activation_dropout':0,
        #     'layernorm_eps':1e-6,
        #     'subln':False,
        #     'use_rms_norm':False,
        #     'use_bias':True,
        # }
        # self.proj_out   = FeedForwardNetwork(**ffm_config)  ###<--- this module is quite hard to train
        self.proj_out = nn.Sequential(
            nn.Linear(config.cross_hidden_size, config.cross_hidden_size, bias=False),
            nn.Tanh(),
            nn.LayerNorm( config.cross_hidden_size, eps=1e-6),
            nn.Linear(config.cross_hidden_size, config.cross_hidden_size),
        )

    def get_distance_qkv(self, distance_state):
        q = self.query_distance(distance_state).unsqueeze(1)# (B, G, 42) -> (B, 1, G, 256)
        k = self.keyes_distance(distance_state).unsqueeze(1)# (B, G, 42) -> (B, 1, G, 256)
        v = self.value_distance(distance_state).unsqueeze(1)# (B, G, 42) -> (B, 1, G, 256)
        return q, k, v

    def get_hidden_state_qkv(self, hidden_state):
        q = self.query_hidden_state(hidden_state) # (B, L, G, 512) -> (B, L, G, 256)#
        k = self.keyes_hidden_state(hidden_state) # (B, L, G, 512) -> (B, L, G, 256)#
        v = self.value_hidden_state(hidden_state) # (B, L, G, 512) -> (B, L, G, 256)#
        return q, k, v
    
    def combine_distance_state_and_hidden_state(self, distance_state, hidden_state):
        B, _, G, D =distance_state.shape
        B, L, G, D =hidden_state.shape
        return (distance_state*hidden_state).view(B*L, G, D)

    def forward(self, distance_state, hidden_state, station_mask, use_flashattn=True):
        B, L, G, D = hidden_state.shape
        
        assert station_mask.shape == (B*L,G), f"please use bool mask with shape ({B*L},{G}), your station mask is {station_mask.shape}"
        assert station_mask.dtype == torch.bool, f"please use bool mask with shape ({B*L},{G}), your station mask is {station_mask.dtype}"
        attention_mask = station_mask#[:,0]
        qd, kd, vd = self.get_distance_qkv(distance_state)
        qh, kh, vh = self.get_hidden_state_qkv(hidden_state)

        o = self.attention(self.combine_distance_state_and_hidden_state(qd,qh), 
                           self.combine_distance_state_and_hidden_state(kd,kh),
                           self.combine_distance_state_and_hidden_state(vd,vh),
                           attention_mask,use_flashattn=use_flashattn) # (B, G, 256)
        o = self.layer_norm(o)
        o = self.proj_out(o) # (B, G, 256)
        return o

    
class DistanceAttention_Beta(_AttentionLayer):
    def build_distance_qkv(self, config):
        self.keyes_distance = nn.Linear( config.distance_system_dim, config.cross_hidden_size, bias=False)
        self.value_distance = nn.Linear( config.distance_system_dim, config.cross_hidden_size, bias=False)

    def build_hidden_state_qkv(self,config,first_layer=True):
        hidden_state_dim = config.Backbone.hidden_size if first_layer else config.cross_hidden_size
        self.query_hidden_state = nn.Linear( hidden_state_dim, config.cross_hidden_size, bias=False)
    
    def get_distance_qkv(self, distance_state):
        q = self.query_distance(distance_state).unsqueeze(1)
        return q, 1, 1

    def get_hidden_state_qkv(self, hidden_state):
        k = self.keyes_hidden_state(hidden_state)
        v = self.value_hidden_state(hidden_state)
        return 1, k, v
    


class DistanceAttention_Gamma(DistanceAttention_Alpha):
    def get_distance_qkv(self, distance_state):
        q = self.query_distance(distance_state).unsqueeze(1)# (B, G, 42) -> (B, 1, G, 256)
        k = self.keyes_distance(distance_state).unsqueeze(1)# (B, G, 42) -> (B, 1, G, 256)
        v = self.value_distance(distance_state).unsqueeze(1)# (B, G, 42) -> (B, 1, G, 256)
        return q, k, v

    def get_hidden_state_qkv(self, hidden_state):
        q = torch.softmax(self.query_hidden_state(hidden_state),-1) # (B, L, G, 512) -> (B, L, G, 256)#
        k = torch.softmax(self.keyes_hidden_state(hidden_state),-1) # (B, L, G, 512) -> (B, L, G, 256)#
        v = torch.softmax(self.value_hidden_state(hidden_state),-1) # (B, L, G, 512) -> (B, L, G, 256)#
        return q, k, v
    

        
from .retnet.modeling_retnet import RMSNorm
class DistanceAttention_Zeta(DistanceAttention_Gamma):
    def build_layer_norm(self, config):
        self.layer_norm         = RMSNorm( config.cross_hidden_size, eps=1e-6) 

class DistanceAttention_Kappa(DistanceAttention_Zeta):
    def build_distance_qkv(self, config):
        [self.query_distance,self.keyes_distance,self.value_distance] = [
            nn.Sequential(nn.Linear( config.distance_system_dim, config.cross_hidden_size, bias=False),
                          nn.Tanh(),
                          RMSNorm( config.cross_hidden_size, eps=1e-6),
                          nn.Linear(config.cross_hidden_size, config.cross_hidden_size),
                          ) for _ in range(3)]

class DistanceSystemDecoder(nn.Module):
    def __init__(self, config:SignalModelConfig, layer_class = DistanceAttention_Alpha):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([layer_class(config, first_layer=True)])
        for _ in range(config.cross_num_hidden_layers-1):
            self.layers.append(layer_class(config, first_layer=False))

    def forward(self, distance_state, hidden_state, vector_state, station_mask,  use_flashattn=True):
        for layer in self.layers:
            hidden_state = layer(distance_state, hidden_state, station_mask, use_flashattn=use_flashattn)
        o = hidden_state
        mask = station_mask.to(o)
        mask = (mask[:,:,None]@mask[:,None])
        o = (o@o.mT)
        o = torch.einsum('BGj,Bjk,BGks->BGs',o, mask, vector_state) # (B, G, 256) @ (B, 256, G) @ (B, G, G) @(B, G, 2) -> (B, G, 2)
        return o

class RetNetGroup:
    def freeze_backbone(self, freeze_mode, mode='train'):
        if freeze_mode is None:return
        if not freeze_mode: return 
        if freeze_mode == 'only_groupbranch':
            for param in self.parameters():
                param.requires_grad = False
            for module in self.modules():
                module.eval()  # Set the module to evaluation mode
            free_layer = self.group_modules
            print0(f"The model is freezen, except group_modules branch !!! ")
            for param in free_layer.parameters():
                param.requires_grad = True
            if mode == 'train':
                for module in free_layer.modules():
                    module.train()  # Set the module to evaluation mode
            self.skip_loss_backward_key.append('distance')
        else:
            raise NotImplementedError(f"freeze mode {freeze_mode} not support")
    
    def compute_vector_combine_distance_and_hidden_state(self, distance_state, hidden_states, vector_state, station_mask,  use_flashattn=True):
        # hidden_states --> (BG, num_layers, 1, 256)
        BG, N, L, D = hidden_states.shape

        #assert L == 1, f"the length of the hidden_state should be 1, but now is {L}"
        station_state = hidden_states # (BG, num_layers, 256)
        B,G, _ = distance_state.shape
        ## recover hidden_state from (B*G, L, D) to (B, G, L, D)
        shaped_state = torch.zeros(B, G, N, L, D).to(station_state)
        shaped_state[station_mask] = station_state

        shaped_state = rearrange(shaped_state, 'B G N L D->N B L G D')#.permute(2, 0, 3, 1, 4) # (num_layers, B,L, G, D) 
        station_mask = repeat(station_mask, 'B G -> (B L) G', L=L)
        vector_state = repeat(vector_state, 'B N M C -> (B L) N M C', L=L)
        vector_predicted = []
        for hidden_state, predictor in zip(shaped_state, self.group_modules['vector_predictor']):
            vector_predicted.append(predictor(distance_state, hidden_state, vector_state, station_mask, use_flashattn=use_flashattn))
        vector_predicted = torch.stack(vector_predicted, dim=-1) # (BL, G, 2, num_layers)

        vector_predicted = self.group_modules['vector_merge'](vector_predicted).squeeze(-1) # (BL, G, 2)
        vector_predicted = rearrange(vector_predicted, '(B L) G C -> B G L C', B=B, L=L, G=G, C=2)
        return vector_predicted  
    
    def formalize_hidden_state(self, outputs:RetNetOutputWithPast):
        if self.config.use_whole_layer_output:
            return torch.stack(outputs.hidden_states,1) #(BG, num_layers, L, 256)
        else:
            return outputs.last_hidden_state.unsqueeze(1) # (BG, 1, L, 256)

    def get_downstream_prediction(self, downstream, distance_expansion=None, vector_state=None, station_mask=None):
        # predicted_vector (B,G,2)
        hidden_state = downstream.pop('group_vector')
        predicted_vector = self.compute_vector_combine_distance_and_hidden_state(distance_expansion, hidden_state, 
                                                                                 vector_state, station_mask, 
                                                                                 use_flashattn=self.config.use_flashattn_in_group_attention)
        #print(predicted_vector.shape)
        if self.config.only_train_multistation:# and self.group_modules.trainning:
            preded = {}
        else:
            preded = self.downstream_prediction(downstream)
            # {key1: (B*G, L, 1), ke2: (B*G, L, 1}
        
        predicted_vector = predicted_vector[station_mask]
        preded['group_vector'] = predicted_vector # (BG, 2)
        
        return preded

    def deal_with_labels(self, labels, preded, station_mask, confidence=None):
        target = {'confidence':confidence, 'station_mask':station_mask}
        for key, val in labels.items():
            if key not in preded:continue
            if len(val.shape) == 1: #(B,)
                raise NotImplementedError(f'it is quite wired that you only have 1D label for group prediction task')
            elif len(val.shape)==2:
                # (B, G)
                target[key] = val[station_mask].unsqueeze(-1) #(B,G) -> (BG,2)
            elif len(val.shape)==3:
                # (B, G, 2)
                if 'vector' in key:
                    target[key] = val[station_mask].unsqueeze(1) #(BG, 1,2)
                else:
                    target[key] = val[station_mask]
            else:
                target[key] = val[station_mask] #(BG, d)
            assert len(target[key]) == len(preded[key]), f"the length of the target[{key}] is {len(target[key])} but the length of the preded[{key}] is {len(preded[key])}"
        return target 

    @torch.inference_mode()
    def generate_next(self, last_state, now_status_seq, now_waveform_seq,now_trend_seq=None,
                                    distance_expansion: Optional[Tuple[torch.FloatTensor]] = None,
                                    vector_state: Optional[Tuple[torch.FloatTensor]] = None,
                                    station_mask: Optional[Tuple[torch.BoolTensor]] = None,):
        self.isinrecurrent_mode = True
        assert now_status_seq.shape[-1] % 3 == 0, f"the length of the sequence should be multiple of 3, but now is {now_status_seq.shape} " #(B,N,L,3)
        last_sequence = last_state['cached_sequence']
        last_seq_len  = last_state['seq_len']
        past_kv       = last_state['past_kv']
        if 'unnormlized_kv' in past_kv: assert past_kv['unnormlized_kv'] is not None, "you may use model.eval to enable the cache"
        if 'prev_key_value' in past_kv: assert past_kv['prev_key_value'] is not None, "you may use model.eval to enable the cache"
        station_mask        = station_mask.bool()
        now_waveform_seq    = now_waveform_seq[station_mask] ## (B, G, L, D) -> (B*G, L, D)

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
    def generate_start_pointing(self, start_status_seq, start_waveform_seq, start_trend_seq=None,
                                    distance_expansion: Optional[Tuple[torch.FloatTensor]] = None,
                                    vector_state: Optional[Tuple[torch.FloatTensor]] = None,
                                    station_mask: Optional[Tuple[torch.BoolTensor]] = None,
                                    ):
        station_mask        = station_mask.bool()
        start_waveform_seq  = start_waveform_seq[station_mask] ## (B, G, L, D) -> (B*G, L, D)
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
    
####### Group Model ############
class RetNetGroupSlide(RetNetGroup,RetNetSlidePred):

    def __init__(self, args: SignalModelConfig):
        super().__init__(args)
        print0(f"====> Notice: The group model so far only design for end-to-end task like DiTingGroup")
        self.group_modules = nn.ModuleDict()
        num_distance_system_layers = args.Backbone.num_hidden_layers if args.use_whole_layer_output else 1
        self.group_modules['vector_predictor'] = nn.ModuleList([DistanceSystemDecoder(args, eval(args.group_decoder_class)) for _ in range(num_distance_system_layers)])
        self.group_modules['vector_merge']     = nn.Linear(num_distance_system_layers, 1, bias=False)
        self.post_init()
    
    
        
    def collect_kernel_output(self, outputs:RetNetOutputWithPast):
        hidden_states = self.formalize_hidden_state(outputs)
        if   self.config.Predictor.merge_token == 'average': fea = hidden_states.mean(-2, keepdims=True) # (BG, num_layers, 1, 256)
        elif self.config.Predictor.merge_token == 'last' :   fea = hidden_states[..., -1:, :] # (BG, num_layers, 1, 256)
        elif self.config.Predictor.merge_token == 'first':   fea = hidden_states[..., 0:1, :] # (BG, num_layers, 1, 256)
        else:
            raise ValueError("merge_token only support average, last, first")
        
        downstream_feature = {}
        for key in self.predictor.keys():
            assert key not in ['findP', 'findS', 'findN']
            downstream_feature[key] = fea[:,-1] ### <<<---only use the last one as the downstream feature, to keep same as pretrain model
        downstream_feature['group_vector'] = fea
        
        return fea,downstream_feature #past_key_values, fea, hidden_states
    
    
    
    def forward(
        self,
        status_seq: Optional[torch.LongTensor] = None,
        waveform_seq: Optional[torch.FloatTensor] = None,
        trend_seq: Optional[torch.FloatTensor] = None,
        distance_expansion: Optional[Tuple[torch.FloatTensor]] = None,
        vector_state: Optional[Tuple[torch.FloatTensor]] = None,
        station_mask: Optional[Tuple[torch.BoolTensor]] = None,
        past_kv: Optional[List[torch.FloatTensor]] = None,
        labels=None,
        get_prediction: Optional[str] = None,
    ):  
    
        assert len(waveform_seq.shape) == 4, f"the waveform_seq should be (B, G, L, D), but now is {waveform_seq.shape}"
        assert waveform_seq is not None, f"disable input_ids, so must provide inputs_embeds"
        assert status_seq is None or len(status_seq.shape) == 3
        B, G, L, _ = waveform_seq.shape
        # (distance_expansion, # (B, G, 2*(G,2)) ## the   di,...,   didj expansion and the 1/di,..., 1/didj expansion
        #  vector_state,       # (B, G, 2)       ## the vector state
        #  station_mask,       # (B, G, G)       ## which station we used
        #  )= distance_matrix_system 
        station_mask = station_mask.bool()
        waveform_seq        = waveform_seq[station_mask] ## (B, G, L, D) -> (B*G, L, D)
        status_seq          = self.deal_with_autoregress_sequence(status_seq)
        inputs_embeds       = self.get_composed_input_embedding(status_seq, waveform_seq,trend_seq=trend_seq)
        inputs_embeds       = self.get_key_token_in_parallel_mode(inputs_embeds) 
        hidden_state, downstream  = self.kernel_forward(inputs_embeds, past_kv=past_kv, use_cache=False, forward_impl='parallel')

        
        if labels is None:
            return SignalOutput(last_hidden_state=hidden_state, prediction=preded)
        else:
            if 'phase' in preded:labels['phase'] = status_seq  
            target = self.deal_with_labels(labels, preded, station_mask=station_mask,confidence=status_seq)
            loss, error_record, prediction = self.evaluate_error(target, preded, get_prediction=get_prediction)
            return SignalOutput(
                loss=loss,
                last_hidden_state=hidden_state,
                error_record=error_record,
                prediction=prediction
            )
    


class RetNetGroupSea(RetNetGroup, RetNetSignalSea):
    def __init__(self, args: SignalModelConfig):
        super().__init__(args)
        print0(f"====> Notice: The group model so far only design for end-to-end task like DiTingGroup")
        self.group_modules = nn.ModuleDict()
        num_distance_system_layers = args.Backbone.num_hidden_layers if args.use_whole_layer_output else 1
        self.group_modules['vector_predictor'] = nn.ModuleList([DistanceSystemDecoder(args, eval(args.group_decoder_class)) for _ in range(num_distance_system_layers)])
        self.group_modules['vector_merge']     = nn.Linear(num_distance_system_layers, 1, bias=False)
        self.post_init()
    
    def collect_kernel_output(self, outputs):
        hidden_states = self.formalize_hidden_state(outputs)
        downstream_feature = {}
        for key in self.predictor.keys():
            downstream_feature[key] = hidden_states[:,-1] #(B, 1, L, D) -> (B,  L, D)
        if not self.config.only_train_multistation:#
            increment     = torch.cat([t.norm(dim=(-2,-1)) if len(t.shape)==5 else t for t in outputs.increment],1)# (B, layer_num*head_num, L)
            increment     = increment.permute(0,2,1)# (B, L, layer_num*head_num)
            downstream_feature['increment'] = increment
        downstream_feature['group_vector'] = hidden_states
        return hidden_states, downstream_feature
    
    def forward(
        self,
        status_seq: Optional[torch.LongTensor] = None,
        waveform_seq: Optional[torch.FloatTensor] = None,
        trend_seq: Optional[torch.FloatTensor] = None,
        distance_expansion: Optional[Tuple[torch.FloatTensor]] = None,
        vector_state: Optional[Tuple[torch.FloatTensor]] = None,
        station_mask: Optional[Tuple[torch.BoolTensor]] = None,
        past_kv: Optional[List[torch.FloatTensor]] = None,
        labels=None,
        get_prediction: Optional[str] = None,
    ):  

        assert len(waveform_seq.shape) == 4, f"the waveform_seq should be (B, G, L, D), but now is {waveform_seq.shape}"
        assert waveform_seq is not None, f"disable input_ids, so must provide inputs_embeds"
        assert status_seq is None or len(status_seq.shape) == 3
        B, G, L, _ = waveform_seq.shape
        # (distance_expansion, # (B, G, 2*(C_G^2)) ## the   di,...,   didj expansion and the 1/di,..., 1/didj expansion
        #  vector_state,       # (B, G, 2)       ## the vector state
        #  station_mask,       # (B, G, G)       ## which station we used
        #  )= distance_matrix_system 
        station_mask        = station_mask.bool()
        waveform_seq        = waveform_seq[station_mask] ## (B, G, L, D) -> (B*G, L, D)
        status_seq          = self.deal_with_autoregress_sequence(status_seq)
        inputs_embeds       = self.get_composed_input_embedding(status_seq, waveform_seq,trend_seq=trend_seq)
        inputs_embeds       = self.get_key_token_in_parallel_mode(inputs_embeds) 
        hidden_state, downstream  = self.kernel_forward(inputs_embeds, past_kv=past_kv, use_cache=False, forward_impl='parallel')
        if 'increment' in downstream:
            increment           = downstream.pop('increment')
        preded = self.get_downstream_prediction(downstream, distance_expansion, vector_state, station_mask)
        if labels is None:
            return SignalOutput(last_hidden_state=hidden_state, prediction=preded)
        else:
            target = self.deal_with_labels(labels, preded, station_mask=station_mask,confidence=status_seq)
            loss, error_record, prediction = self.evaluate_error(target, preded, get_prediction=get_prediction)
            if not self.config.only_train_multistation  and len(downstream)>0:
                assert 'status' in labels, f"the labels should contain the status, but now is {labels.keys()}"
                if get_prediction:
                    prediction['activation'] = {}
                    prediction['activation']['pred'] = increment.float().detach().cpu().numpy() ## < --- too large for save
                
                increment, true_label = self.build_status_pair(increment,target['status'])
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
    

