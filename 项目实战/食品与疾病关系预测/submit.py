import torch
from torch.utils.data import DataLoader


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

import pandas as pd

from model import MyModel


def submit(model: torch.nn.Module, test_dataloader):
    test_sub = pd.read_csv('./data/初赛A榜测试集/preliminary_a_submit_sample.csv')
    pred_prob = []
    for i, batch in tqdm(enumerate(test_dataloader), desc="评估中...", total=len(test_dataloader)):
        batch = {k: v.cuda() for k, v in batch.items()}
        logits = model(**batch)["logits"]
        pred_prob = pred_prob + logits.tolist()
    # 没有什么机智的后处理
    test_sub['related_prob'] = pred_prob
    test_sub.to_csv('./submit/test04.csv', index=False)


from data_process import get_data, NNDataset

if __name__ == '__main__':
    train, test = get_data()
    test_dataset = NNDataset(test, train_mode=False)
    path = "/home/zut/liuyu/code_dir/ml-nlp/项目实战/食品与疾病关系预测/save2/best_model_91.43377576256412.pt"
    model = torch.load(open(path, "rb"))
    test_dataloader = DataLoader(test_dataset, batch_size=1024, num_workers=1, collate_fn=fc)
    model.eval()
    with torch.no_grad():
        submit(model, test_dataloader)
