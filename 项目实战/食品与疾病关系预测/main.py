import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from model import MyModel


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


def evel(model: nn.Module, dev_dataloader):
    model.eval()
    f1_list = []
    auc_list = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dev_dataloader), desc="评估中...", total=len(dev_dataloader)):
            batch = {k: v.cuda() for k, v in batch.items()}
            label = batch["label"]
            logits = model(**batch)["logits"]
            pre_idx = torch.ge(logits, 0.5)
            logits = np.array(logits.cpu())
            pre_idx = np.array(pre_idx.cpu()).astype(int)
            label = np.array(label.cpu()).astype(int)
            f1 = f1_score(label, pre_idx) * 100
            auc = roc_auc_score(label, logits) * 100
            f1_list.append(f1)
            auc_list.append(auc)
        f1 = np.mean(f1_list)
        auc = np.mean(auc_list)
        res = (f1 + auc) / 2
        return {"f1": f1, "auc": auc, "res": res}


from data_process import NNDataset, get_data

if __name__ == '__main__':
    train, test = get_data()
    train_dataset = NNDataset(train)
    # 根据标签进行分层抽样
    folds = 10
    期望运行的轮次 = 50
    epochs = 期望运行的轮次 // folds
    max_sore = 0
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    model = MyModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epoch = 0
    iters = 0
    for _ in range(epochs):
        for fold, (train_idx, dev_idx) in enumerate(skf.split(X=train_dataset.data, y=train_dataset.target)):
            epoch = epoch + 1
            print(f'Fold {fold + 1}')
            train = Subset(train_dataset, train_idx)
            dev = Subset(train_dataset, dev_idx)
            train_dataloader = DataLoader(train, batch_size=256, shuffle=True, num_workers=1, collate_fn=fc)
            dev_dataloader = DataLoader(dev, batch_size=256, shuffle=True, num_workers=1, collate_fn=fc)

            # # torch 拆分数据集
            # train_num = len(train_dataset)
            # train_sample = int(train_num * 0.8)
            # # 前0.8 数据   和 后 0.2 数据
            # train, dev = random_split(train_dataset, [train_sample, train_num - train_sample])
            # train_dataloader = DataLoader(train, batch_size=256, shuffle=True, num_workers=1, collate_fn=fc)
            # dev_dataloader = DataLoader(dev, batch_size=256, shuffle=True, num_workers=1, collate_fn=fc)
            model.train()
            for batch in train_dataloader:
                batch = {k: v.cuda() for k, v in batch.items()}
                loss = model(**batch)["loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iters % 50 == 0:
                    print("epoch:{}  batch:{}  loss:{:.4}".format(epoch, iters, loss.item()))
                iters = iters + 1
            score = evel(model, dev_dataloader)
            print(score)
            if score["res"] > max_sore:
                max_sore = score["res"]
                if epoch >= 10:
                    torch.save(model, f"./save2/best_model_{max_sore}.pt")
                    print(f"最优模型已保存...  res:{max_sore}")
