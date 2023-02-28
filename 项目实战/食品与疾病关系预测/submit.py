import torch
from torch.utils.data import DataLoader

from data_process import test_dataset


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


from tqdm import tqdm

import numpy as np
import pandas as pd


def submit(model: torch.nn.Module, test_dataloader):
    test_sub = pd.read_csv('./data/初赛A榜测试集/preliminary_a_submit_sample.csv')
    pred_prob = []
    for i, batch in tqdm(enumerate(test_dataloader), desc="评估中...", total=len(test_dataloader)):
        batch = {k: v.cuda() for k, v in batch.items()}
        logits = model(**batch)["logits"] + 0.2
        pred_prob = pred_prob + logits.tolist()
    # 没有什么机智的后处理
    test_sub['related_prob'] = pred_prob
    test_sub.to_csv('./submit/test01.csv', index=False)


import torch.nn as nn


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
        v = torch.dropout(v, p=0.3, train=self.training)
        logits = self.fc(v)
        loss_fct = nn.BCELoss()
        logits = logits.squeeze(dim=-1)
        loss = loss_fct(logits, label.float())
        return {"loss": loss, "logits": logits}


if __name__ == '__main__':
    path = "/home/zut/liuyu/code_dir/ml-nlp/项目实战/食品与疾病关系预测/save/best_model_86.04036045413973.pt"
    model = torch.load(open(path, "rb"))
    test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=1, collate_fn=fc)
    model.eval()
    with torch.no_grad():
        submit(model, test_dataloader)
