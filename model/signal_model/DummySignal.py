import torch
import torch.nn as nn
from typing import Dict, Optional
from einops.layers.torch import Rearrange
from .SignalModel import SignalMagTimeDis


class DummyModel(nn.Module, SignalMagTimeDis):
    def __init__(self, args, downstream_pool):
        super().__init__()
        self.min_used_length = 100
        self.sequence_embedding = nn.Sequential(
            nn.Linear(3, args.hidden_size, bias=False),
            nn.Tanh()
        )
        layers = []
        for _ in range(2):
            layers.append(nn.Linear(args.hidden_size,
                          args.hidden_size, bias=True))
            layers.append(nn.Tanh())
            layers.append(nn.LayerNorm(args.hidden_size))

        layers.extend([
            Rearrange('B L D -> B D L'),
            nn.Linear(args.max_length, 1, bias=False),
            Rearrange('B D L -> B L D'),

        ])

        self.backbone = nn.Sequential(*layers)
        self.build_downstring_task(args, downstream_pool)

    def get_composed_input_embedding(self, status_seq, waveform_seq):
        enc_out = self.sequence_embedding(waveform_seq)
        return enc_out

    def deal_with_autoregress_sequence(self, status_seq):
        return status_seq

    def kernel_forward(self, x):
        x = self.backbone(x)  # (B, L, D)
        return None, x

    def forward(
        self,
        status_seq: Optional[torch.LongTensor] = None,
        waveform_seq: Optional[torch.FloatTensor] = None,
        labels=None,
        get_prediction: Optional[bool] = None,
    ):
        status_seq = self.deal_with_autoregress_sequence(status_seq)
        inputs_embeds = self.get_composed_input_embedding(
            status_seq, waveform_seq)
        _, hidden_states = self.kernel_forward(inputs_embeds)
        preded = self.downstream_prediction(hidden_states)

        target = {}

        if labels:
            for key, val in labels.items():
                if len(val.shape) > 1 and self.min_used_length and val.shape[-1] > self.min_used_length:
                    # (B, L) ==> (B,100)
                    target[key] = val[:, self.min_used_length:]
                elif len(val.shape) == 1:
                    target[key] = val.unsqueeze(1)  # ==> (B,1)
                else:
                    target[key] = val
            loss, error_record, prediction = self.evaluate_error(
                target, preded, get_prediction=get_prediction)
        else:
            loss = error_record = prediction = None
        return {'loss': loss, 'error_record': error_record, 'prediction': prediction}
