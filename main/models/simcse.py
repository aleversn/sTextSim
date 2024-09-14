import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, BertPreTrainedModel, RobertaPreTrainedModel
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.utils import ModelOutput


def load_from_pretrained(self, from_pretrained):
    pretrained_state_dict = torch.load(os.path.join(
        from_pretrained, 'pytorch_model.bin'), map_location='cpu')
    bert_state_dict = self.model.state_dict()
    for key, val in pretrained_state_dict.items():
        exists = False
        for _key in bert_state_dict:
            if key.endswith(_key):
                exists = True
                break
        if exists:
            bert_state_dict[_key] = val
    self.model.load_state_dict(bert_state_dict, strict=False)
    # for key, val in pretrained_state_dict.items():
    #     for name, module in self.model.named_modules():
    #         if name not in ['', 'embeddings']:
    #             if key.endswith(name + '.weight'):
    #                 module.weight.data = val
    #             elif key.endswith(name + '.bias'):
    #                 module.bias.data = val
    self.load_state_dict(pretrained_state_dict, strict=False)


def cl_init(self, from_pretrained, pooler_type, hard_negative_weight, temp):
    self.model = AutoModel.from_pretrained(from_pretrained)
    self.pooler_type = pooler_type
    self.hard_negative_weight = hard_negative_weight
    self.temp = temp

    self.pooler_type = self.pooler_type
    self.pooler = Pooler(self.pooler_type)
    if self.pooler_type == "cls":
        self.mlp = MLPLayer(self.config)
    self.sim = Similarity(temp=self.temp)
    # self.init_weights()
    load_from_pretrained(self, from_pretrained)


def cl_forward(self,
               input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               labels=None,
               output_attentions=None,
               output_hidden_states=None,
               return_dict=None):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    batch_size = input_ids.size(0)

    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)
    is_direct_output = len(input_ids.shape) == 2

    # Flatten input for encoding
    input_ids = input_ids.view(
        (-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view(
        (-1, attention_mask.size(-1)))  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(
            (-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

    # Get raw embeddings
    outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if self.pooler_type in [
            'avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    if is_direct_output:
        pooler_output = self.pooler(attention_mask, outputs)

        return SimCSEOutput(
            loss=None,
            logits=None,
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Pooling
    pooler_output = self.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view(
        (batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if self.pooler_type == "cls":
        pooler_output = self.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # 原始是bs * bs 最终得到bs * 1的数值相乘矩阵, 这里通过改成bs * 1 * 1 * bs得到bs * bs的数值相乘矩阵
    cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(self.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = self.hard_negative_weight
        # 画一个前cos_sim.size(-1)列为0, 后z1_z3_cos.size(-1)列对角线为0 + z3_weight的矩阵
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [
                0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(self.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SimCSEOutput(
        loss=loss,
        logits=cos_sim,
        last_hidden_state=outputs.last_hidden_state,
        pooler_output=pooler_output,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class SimCSE(BertPreTrainedModel):

    def __init__(self, from_pretrained, pooler_type, hard_negative_weight=0, temp=0.05):
        self.config = AutoConfig.from_pretrained(from_pretrained)
        super().__init__(self.config)

        cl_init(self, from_pretrained, pooler_type, hard_negative_weight, temp)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **custom_args):

        return cl_forward(self,
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          position_ids=position_ids,
                          head_mask=head_mask,
                          inputs_embeds=inputs_embeds,
                          labels=labels,
                          output_attentions=output_attentions,
                          output_hidden_states=output_hidden_states,
                          return_dict=return_dict)


class SimCSERoberta(RobertaPreTrainedModel):
    def __init__(self, from_pretrained, pooler_type, hard_negative_weight=0, temp=0.05):
        self.config = AutoConfig.from_pretrained(from_pretrained)
        super().__init__(self.config)

        cl_init(self, from_pretrained, pooler_type, hard_negative_weight, temp)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **custom_args):

        return cl_forward(self,
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=None,
                          position_ids=position_ids,
                          head_mask=head_mask,
                          inputs_embeds=inputs_embeds,
                          labels=labels,
                          output_attentions=output_attentions,
                          output_hidden_states=output_hidden_states,
                          return_dict=return_dict)


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 *
                             attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 *
                             attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


@dataclass
class SimCSEOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
