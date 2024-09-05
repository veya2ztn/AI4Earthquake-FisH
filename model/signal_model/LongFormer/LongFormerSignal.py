import torch
from transformers import LongformerForTokenClassification
from transformers.models.longformer.configuration_longformer import LongformerConfig
from transformers.models.longformer.modeling_longformer import LongformerEmbeddings
from transformers.models.longformer.modeling_longformer import *
from typing import Dict, Optional, List,Tuple
from .SignalModel import SignalMagTimeDis,FastABS

@dataclass
class SignalOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    prediction: Optional[Dict[str, torch.Tensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None
    error_record: Optional[Dict[str, torch.Tensor]] = None

class QuakeEmbedding(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout   = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        embeddings = inputs_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LongformerQuakeEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings       = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Linear(3, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout   = nn.Dropout(config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()
        assert token_type_ids is not None

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class LongSignalBase(LongformerPreTrainedModel, SignalMagTimeDis):
    

    def kernel_forward(self, input_ids,attention_mask,global_attention_mask,head_mask,
                       token_type_ids, position_ids, output_attentions, output_hidden_states, return_dict):
        assert input_ids is not None
        assert token_type_ids is not None
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0] #focus on the CLS token
        return outputs, hidden_states

    

    def deal_with_autoregress_sequence(self, status_seq):
        #status_seq       = torch.nn.functional.pad(status_seq,(1,0))[:,:-1].long() #(B, L) -> (B, L+1) -> (B, L)
        status_seq = torch.roll(status_seq, 1, 1)
        status_seq[:, 0] = self.class_token
        return status_seq
    

    def forward(
        self,
        status_seq: Optional[torch.Tensor],
        waveform_seq: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        get_prediction: Optional[bool] = None,
    ) -> Union[Tuple, LongformerMultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `status_seq` above)
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if global_attention_mask is None:
            logger.info("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(status_seq)
            # global attention on cls token
            global_attention_mask[:, 0] = 1
        status_seq = self.deal_with_autoregress_sequence(status_seq)
     
        outputs, hidden_states = self.kernel_forward(status_seq, attention_mask, global_attention_mask, head_mask,
                                                     waveform_seq, position_ids, output_attentions, output_hidden_states, return_dict)

        preded = self.downstream_prediction(hidden_states)

        loss = error_record = prediction = None
        if labels is not None:
            target = {}
            for key, val in labels.items():
                if len(val.shape) > 1 and self.min_used_length and val.shape[-1] > self.min_used_length:
                    target[key] = val[..., self.min_used_length:]
                elif len(val.shape) == 1:
                    target[key] = val.unsqueeze(1)  # ==> (B,1)
                else:
                    target[key] = val
            loss, error_record, prediction = self.evaluate_error(
                target, preded, get_prediction=get_prediction)

        return SignalOutput(
            loss=loss,
            logits=preded['phase'] if 'phase' in preded else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
            prediction=prediction,
            error_record=error_record,
        )

class LongPrediction(LongSignalBase):
    # def __init__(self, config):
    #     super().__init__(config)
    #     self.embeddings = None
    #     self.embedding_merge = None
    def __init__(self, config, downstream_pool):
        super().__init__(config)
        self.start_tokeon = config.bos_token_id
        self.min_used_length = 100
        self.longformer = LongformerModel(config)
        self.longformer.embeddings = LongformerQuakeEmbeddings(config)
        # self.embeddings          = nn.Embedding(config.vocab_size, config.hidden_size)
        self.build_downstring_task(config, downstream_pool)
        assert config.vocab_size == 4
        self.class_token = 3
        self.post_init()
    


class LongSignalSequence(LongSignalBase):
    def __init__(self, config, downstream_pool):
        super().__init__(config)
        self.start_tokeon          = config.bos_token_id
        self.min_used_length       = 100
        self.longformer            = LongformerModel(config)
        self.longformer.embeddings = LongformerQuakeEmbeddings(config)
        # self.embeddings          = nn.Embedding(config.vocab_size, config.hidden_size)
        self.build_downstring_task(config, downstream_pool)
        assert config.vocab_size == 4
        self.class_token = 3
        self.post_init()

    def kernel_forward(self, input_ids,attention_mask,global_attention_mask,head_mask,
                       token_type_ids, position_ids, output_attentions, output_hidden_states, return_dict):
        assert input_ids is not None
        assert token_type_ids is not None
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0][:,0:1,:] #focus on the CLS token
        return outputs, hidden_states

    

    