from data_process import train_dataset, test_dataset
import torch
# 构建数据集
from torch.utils.data import DataLoader


def fc(batch):
    food_feat = [item[0] for item in batch]
    dis_feat1 = [item[0] for item in batch]
    dis_feat2 = [item[0] for item in batch]
    dis_feat3 = [item[0] for item in batch]
    food_feat = torch.tensor(food_feat)
    dis_feat1 = torch.tensor(food_feat)
    dis_feat2 = torch.tensor(food_feat)
    dis_feat3 = torch.tensor(food_feat)


if __name__ == '__main__':
    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=1, collate_fn=fc)
    for batch in train_dataloader:
        print(batch)
        break
