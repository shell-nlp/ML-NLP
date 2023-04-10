import torch.nn as nn
from transformers import BertModel


class Berts(nn.Module):
    def __init__(self, check_point, num_labels=16):
        super(Berts, self).__init__()
        self.bert = BertModel.from_pretrained(check_point)
        for param in self.bert.parameters():
            param.requires_grad = False
        config = self.bert.config
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.lstm = nn.LSTM(config.hidden_size, config.hidden_size,
        #                     batch_first=True, bidirectional=True)
        # self.relu = nn.ReLU()

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels=None,
                ):
        outputs = self.bert(input_ids,
                                    attention_mask,
                                    token_type_ids,)
        encoder_out = outputs[1]
        logits = self.classifier(encoder_out)
        # output, (ht, hc) = self.lstm(self.dropout(encoder_out))
        # ht = self.relu(ht.view(encoder_out.size(0), -1))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {
            "logits": logits,
            "loss": loss
        }
