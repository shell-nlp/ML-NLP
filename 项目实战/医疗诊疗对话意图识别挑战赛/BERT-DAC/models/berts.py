import torch.nn as nn
from transformers import BertModel


class Berts(nn.Module):
    def __init__(self, check_point, num_labels=16):
        super(Berts).__init__()
        self.bert = BertModel.from_pretrained(check_point)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self,batch):
        outputs = self.bert(**batch)
        
