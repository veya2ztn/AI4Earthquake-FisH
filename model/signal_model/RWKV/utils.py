
import torch
import transformers


def rwkv_linear_attention_faster(time_decay, time_first, key, value, state=None, return_state=False, indices_system=None):
    assert state is None
    assert not return_state
    time_decay = -torch.exp(time_decay)  # (1,D)
    B, S, D = key.size()
    if indices_system is None:
        rows = torch.arange(S).repeat(S, 1).transpose(0, 1)
        cols = torch.arange(S).repeat(S, 1)
        diag = torch.eye(S).bool()
        indices_counts = torch.arange(S-1).float()
        mask = (rows > cols).to(key.device)
        indices_matrix = indices_counts[(rows - cols - 1)[mask]].to(key.device)
    else:
        rows = indices_system['rows']
        cols = indices_system['cols']
        diag = indices_system['diag']
        indices_matrix = indices_system['indices_matrix']

    logits_matrix = -torch.empty(B, S, S, D).fill_(torch.inf).to(key.device)

    #print(f"{logits_matrix.max()} {indices_matrix.max()} {time_decay.max()} {key.max()} {time_first.max()}")
    mask = (rows > cols)

    logits_matrix[:, mask, :] = torch.einsum(
        's,d->sd', indices_matrix, time_decay)
    logits_matrix[:, diag, :] = 0
    mask = (rows >= cols)
    logits_matrix[:, mask, :] += key[:, cols[mask]]
    logits_matrix[:, diag, :] += time_first

    logits_matrix = logits_matrix - logits_matrix.max(2, keepdims=True)[0]
    logits_matrix = logits_matrix.softmax(2)

    output = torch.einsum('bsld, bld->bsd', logits_matrix, value)
    return output, None


def replace_rwkv_attn_with_faster():
    def forward(self, hidden, state=None, use_cache=False):
        assert state is None
        assert not use_cache
        receptance, key, value, state = self.extract_key_value(
            hidden, state=state)
        layer_state = None
        B, S, D = key.size()
        if not hasattr(self, 'indices_system'):
            self.indices_system = {}
        if S not in self.indices_system:
            self.indices_system[S] = {}
            rows = torch.arange(S).repeat(S, 1).transpose(0, 1)
            cols = torch.arange(S).repeat(S, 1)
            diag = torch.eye(S).bool()
            indices_counts = torch.arange(S-1).float()
            mask = (rows > cols)
            indices_matrix = indices_counts[(rows - cols - 1)[mask]]
            self.indices_system[S]['rows'] = rows
            self.indices_system[S]['cols'] = cols
            self.indices_system[S]['diag'] = diag
            self.indices_system[S]['indices_matrix'] = indices_matrix.to(
                key.device)
        rwkv, layer_state = rwkv_linear_attention_faster(
            self.time_decay,
            self.time_first,
            key,
            value,
            state=layer_state,
            return_state=use_cache, indices_system=self.indices_system[S]
        )

        return self.output(receptance * rwkv), None
    transformers.models.rwkv.modeling_rwkv.RwkvSelfAttention.forward = forward
