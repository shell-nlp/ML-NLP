import torch.nn as nn
import torch


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.food = nn.Sequential(
            nn.Linear(212, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.Dropout(0.3)
        )
        self.feat1 = nn.Sequential(
            nn.Linear(128, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.Dropout(0.3)
        )
        self.feat2 = nn.Sequential(
            nn.Linear(128, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.Dropout(0.3)
        )
        self.feat3 = nn.Sequential(
            nn.Linear(128, 768),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, food, feat1, feat2, feat3, label):
        f = self.food(food)
        f1 = self.feat1(feat1)
        f2 = self.feat2(feat2)
        f3 = self.feat3(feat3)
        # v = f + f1 + f2 + f3
        v = torch.cat((f, f1, f2, f3), dim=-1)
        v = torch.dropout(v, p=0.3, train=self.training)
        logits = self.fc(v)
        loss_fct = nn.BCELoss()
        logits = logits.squeeze(dim=-1)
        loss = loss_fct(logits, label.float())
        return {"loss": loss, "logits": logits}
