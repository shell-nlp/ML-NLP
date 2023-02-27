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
    label = [int(item[4]) for item in batch]

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

    def forward(self, food, feat1, feat2, feat3, label):
        f = self.food(food)
        f1 = self.feat1(feat1)
        f2 = self.feat2(feat2)
        f3 = self.feat3(feat3)
        v = f + f1 + f2 + f3
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(v, label)
        return {"loss": loss}


def eve():
    pass


from sklearn.model_selection import StratifiedKFold

# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2023)

if __name__ == '__main__':
    epochs = 10
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=1, collate_fn=fc)
    model = MyModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
    for epoch in range(epochs):
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = model(**batch)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print("epoch:{}  batch:{}  loss:{:.4}".format(epoch, i, loss.item()))
