import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoConfig, PreTrainedModel
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.utils import ModelOutput


class SBert(PreTrainedModel):

    def __init__(self, from_pretrained, max_seq_len=512, fct_loss='MSELoss'):
        self.config = AutoConfig.from_pretrained(from_pretrained)
        super().__init__(self.config)
        self.max_seq_len = max_seq_len
        self.fct_loss = fct_loss
        self.model = BertModel.from_pretrained(
            from_pretrained, config=self.config)

        self.hidden_size = self.config.hidden_size
        self.softmax = nn.Sequential(
            nn.Linear(self.hidden_size * 3, 2),
            nn.Softmax(dim=-1)
        )
        self.cosine_score_transformation = nn.Identity(torch.cosine_similarity)
        self.load_from_pretrained(from_pretrained)

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
        self.load_state_dict(pretrained_state_dict, strict=False)

    def pooling(self, x, mask, mode='mean'):
        # input: batch_size * seq_len * hidden_size
        cls_token = x[:, 0, :]
        output_vectors = []
        if mode == 'cls':
            output_vectors.append(cls_token)
        elif mode == 'max':
            input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
            x[input_mask_expanded == 0] = -1e9
            max_over_timer = torch.max(x, 1)[0]
            output_vectors.append(max_over_timer)
        elif mode == 'mean':
            input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
            sum_embeddings = torch.sum(x * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 1)
        # print('cls_token', cls_token.shape, 'input_mask_expanded', input_mask_expanded.shape, 'sum_mask', sum_mask.shape)
        # print(0 / 0)
        return output_vector

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
                **args):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        max_seq_len = self.max_seq_len // 2
        if 'fct_loss' in args:
            if self.fct_loss == 'BCELoss':
                fct_loss = nn.BCELoss()
            elif self.fct_loss == 'CrossEntropyLoss':
                fct_loss = nn.CrossEntropyLoss()
            elif self.fct_loss == 'MSELoss':
                fct_loss = nn.MSELoss()
        else:
            fct_loss = nn.MSELoss()
        x1, x2 = input_ids[:, :max_seq_len], input_ids[:, max_seq_len:]
        mask1, mask2 = attention_mask[:,
                                      :max_seq_len], attention_mask[:, max_seq_len:]
        if token_type_ids is not None:
            tid1, tid2 = token_type_ids[:,
                                        :max_seq_len], token_type_ids[:, max_seq_len:]
        else:
            tid1, tid2 = None, None

        if position_ids is not None:
            pid1, pid2 = position_ids[:,
                                      :max_seq_len], position_ids[:, max_seq_len:]
        else:
            pid1, pid2 = None, None

        em1 = self.model(
            x1,
            attention_mask=mask1,
            token_type_ids=tid1,
            position_ids=pid1,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True
        )[0]
        em2 = self.model(
            x2,
            attention_mask=mask2,
            token_type_ids=tid2,
            position_ids=pid2,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True
        )[0]

        u = self.pooling(em1, mask1)
        v = self.pooling(em2, mask2)
        similarity = self.cosine_score_transformation(
            torch.cosine_similarity(u, v))

        loss = fct_loss(similarity, labels.float())

        return SBERTOutput(
            loss=loss,
            logits=similarity
        )

    def get_model(self):
        return self.model


@dataclass
class SBERTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
