import torch
import torch.nn as nn
from transformers import BertModel


class Berts(nn.Module):
    def __init__(self, check_point, num_labels=16):
        super(Berts, self).__init__()
        self.bert = BertModel.from_pretrained(check_point)
        for param in self.bert.parameters():
            param.requires_grad = True
        config = self.bert.config
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.hidden_size*4, num_labels)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels=None,
                ):
        outputs = self.bert(input_ids,
                            attention_mask,
                            token_type_ids, output_hidden_states=True)
        # encoder_out = outputs.last_hidden_state
        # text_cls = outputs.pooler_output
        hidden4layer = outputs.hidden_states[-4:]  # 后四层的隐状态
        cated = torch.cat(hidden4layer, dim=-1)  # []
        text_cls = cated[:, 0, :]
        text_cls = self.dropout(text_cls)
        logits = self.classifier(text_cls)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {
            "logits": logits,
            "loss": loss
        }
