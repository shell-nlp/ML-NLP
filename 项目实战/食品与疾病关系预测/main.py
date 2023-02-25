from data_process import train_dataset, test_dataset
import torch
# 构建数据集
from torch.utils.data import DataLoader
import torch.nn as nn


def fc(batch):
    food_feat = [item[0] for item in batch]
    dis_feat1 = [item[1] for item in batch]
    dis_feat2 = [item[2] for item in batch]
    dis_feat3 = [item[3] for item in batch]
    label = [item[4] for item in batch]
    food_feat = torch.tensor(food_feat)
    dis_feat1 = torch.tensor(dis_feat1)
    dis_feat2 = torch.tensor(dis_feat2)
    dis_feat3 = torch.tensor(dis_feat3)
    label = torch.tensor(label)

    return {"food": food_feat, "feat1": dis_feat1, "feat2": dis_feat2, "feat3": dis_feat3, "label": label}


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.food = torch.nn.Sequential(
            nn.Linear(212, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.Dropout(0.3)
        )
        self.feat1 = torch.nn.Sequential(
            nn.Linear(128, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.Dropout(0.3)
        )
        self.feat2 = torch.nn.Sequential(
            nn.Linear(128, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.Dropout(0.3)
        )
        self.feat3 = torch.nn.Sequential(
            nn.Linear(128, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.Dropout(0.3)
        )
        self.fc = torch.nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Dropout(0.3)
        )

        def forward(food, feat1, feat2, feat3, label):
            f = self.food(food)
            f1 = self.food(feat1)
            f2 = self.food(feat2)
            f3 = self.food(feat3)
            v = f + f1 + f2 + f3
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(v, label)
            return {"loss": loss}


if __name__ == '__main__':
    epochs = 10
    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=1, collate_fn=fc)
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0)
    for epoch in range(epochs):
        for batch in train_dataloader:
            loss = model(**batch)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
