import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from transformers import set_seed

seed = 520

set_seed(seed)


# 降维
def 降维(feats, nfeats=64):
    pca = PCA(n_components=nfeats, random_state=seed)
    new_feats = pca.fit_transform(feats)

    return new_feats


def get_data():
    #                                                                   n_sample   feature
    disease_feature1 = pd.read_csv('./data/训练集/disease_feature1.csv')  # 220      997
    disease_feature2 = pd.read_csv('./data/训练集/disease_feature2.csv')  # 301      318
    disease_feature3 = pd.read_csv('./data/训练集/disease_feature3.csv')  # 392      1454
    # 降维可以使用其他方式  暂时使用PCA
    new_feat1 = 降维(disease_feature1.iloc[:, 1:], nfeats=128)
    new_feat2 = 降维(disease_feature2.iloc[:, 1:], nfeats=128)
    new_feat3 = 降维(disease_feature3.iloc[:, 1:], nfeats=128)

    feat1 = pd.DataFrame(new_feat1)
    feat1['disease_id'] = disease_feature1.disease_id

    feat2 = pd.DataFrame(new_feat2)
    feat2['disease_id'] = disease_feature2.disease_id

    feat3 = pd.DataFrame(new_feat3)
    feat3['disease_id'] = disease_feature3.disease_id

    # 数据读取
    test_food = pd.read_csv('./data/初赛A榜测试集/preliminary_a_food.csv')  # 212
    test_sub = pd.read_csv('./data/初赛A榜测试集/preliminary_a_submit_sample.csv')
    train_food = pd.read_csv('./data/训练集/train_food.csv')  # 食物特征    212
    train_answer = pd.read_csv('./data/训练集/train_answer.csv')  # 食物和疾病关系
    # 只需要找 train_answer中对于的标签  所以都用  left
    train = train_answer. \
        merge(train_food, on='food_id', how='left'). \
        merge(feat1, on='disease_id', how='left'). \
        merge(feat2, on='disease_id', how='left'). \
        merge(feat3, on='disease_id', how='left')
    test = test_sub. \
        merge(test_food, on='food_id', how='left'). \
        merge(feat1, on='disease_id', how='left'). \
        merge(feat2, on='disease_id', how='left'). \
        merge(feat3, on='disease_id', how='left')

    train = train.fillna(0)
    test = test.fillna(0)
    return train, test


class NNDataset(Dataset):
    def __init__(self, df, train_mode=True):
        super(NNDataset, self).__init__()
        self.train_mode = train_mode
        self.df = df
        self.data = df.iloc[:, 3:]
        if train_mode:
            self.target = df["related"]
        else:
            self.target = df["related_prob"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx].values
        food_feat = sample[3:3 + 212].astype(np.float32)
        dis_feat1 = sample[3 + 212:3 + 212 + 128].astype(np.float32)
        dis_feat2 = sample[3 + 212 + 128:3 + 212 + 128 * 2].astype(np.float32)
        dis_feat3 = sample[3 + 212 + 128 * 2:].astype(np.float32)
        label = np.array(sample[2], dtype=np.float32)
        return food_feat, dis_feat1, dis_feat2, dis_feat3, label
