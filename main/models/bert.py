import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, PreTrainedModel, AutoModel
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.utils import ModelOutput


class Bert(PreTrainedModel):

    def __init__(self, from_pretrained, num_labels=2, fct_loss='BCELoss'):
        self.config = AutoConfig.from_pretrained(from_pretrained)
        super().__init__(self.config)
        self.fct_loss = fct_loss
        self.num_labels = num_labels
        self.model = AutoModel.from_pretrained(from_pretrained)
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
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

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            if self.fct_loss == 'BCELoss':
                fct_loss = nn.BCELoss()
                positive_logits = F.softmax(logits, dim=1)[:, 1]
                loss = fct_loss(positive_logits, labels.float())
            elif self.fct_loss == 'CrossEntropyLoss':
                fct_loss = nn.CrossEntropyLoss()
                loss = fct_loss(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.fct_loss == 'MSELoss':
                fct_loss = nn.MSELoss()
                if self.num_labels == 1:
                    loss = fct_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = fct_loss(logits, labels)
        else:
            loss = None

        return BERTNLIOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


@dataclass
class BERTNLIOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
