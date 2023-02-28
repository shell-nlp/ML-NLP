from data_process import train_dataset, test_dataset
import torch
# 构建数据集
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np


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
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, food, feat1, feat2, feat3, label):
        f = self.food(food)
        f1 = self.feat1(feat1)
        f2 = self.feat2(feat2)
        f3 = self.feat3(feat3)
        v = f + f1 + f2 + f3
        logits = self.fc(v)
        loss_fct = nn.BCELoss()
        logits = logits.squeeze(dim=-1)
        loss = loss_fct(logits, label.float())
        return {"loss": loss, "logits": logits}


from torch.utils.data import random_split
from sklearn.metrics import f1_score, accuracy_score


def evel(model: nn.Module, dev_dataloader):
    model.eval()
    acc_list = []
    f1_list = []
    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            label = batch["label"]
            logits = model(**batch)["logits"]
            pre_idx = torch.ge(logits, 0.5)
            pre_idx = np.array(pre_idx.cpu()).astype(int)
            label = np.array(label.cpu()).astype(int)
            acc = accuracy_score(label, pre_idx) * 100
            f1 = f1_score(label, pre_idx) * 100
            acc_list.append(acc)
            f1_list.append(f1)
        acc = np.mean(acc)
        f1 = np.mean(f1)
        return acc, f1


if __name__ == '__main__':
    epochs = 10
    train_num = len(train_dataset)
    train_sample = int(train_num * 0.8)
    # 前0.8 数据   和 后 0.2 数据
    train, dev = random_split(train_dataset, [train_sample, train_num - train_sample])
    train_dataloader = DataLoader(train, batch_size=512, shuffle=True, num_workers=1, collate_fn=fc)
    dev_dataloader = DataLoader(dev, batch_size=512, shuffle=True, num_workers=1, collate_fn=fc)
    model = MyModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = model(**batch)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print("epoch:{}  batch:{}  loss:{:.4}".format(epoch, i, loss.item()))
        acc, f1 = evel(model, dev_dataloader)
        print("epoch:{}  acc:{:.4}  f1:{:.4}".format(epoch, acc, f1))
