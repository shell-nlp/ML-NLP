import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, precision_recall_curve
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

SEED = 520
SAVE_PATH = 'model'

cat_feats = []

DATA_PATH = '../data'
train_food = pd.read_csv(os.path.join(DATA_PATH, '训练集', 'train_food.csv'))  # 事物特征
train_answer = pd.read_csv(os.path.join(DATA_PATH, '训练集', 'train_answer.csv'))  # 关系
# 加载三个疾病特征
disease_feature1 = pd.read_csv(os.path.join(DATA_PATH, '训练集', 'disease_feature1.csv'))
disease_feature2 = pd.read_csv(os.path.join(DATA_PATH, '训练集', 'disease_feature2.csv'))
disease_feature3 = pd.read_csv(os.path.join(DATA_PATH, '训练集', 'disease_feature3.csv'))
# ----------------------------------------加载测试数据集------开始-------------------------------------
testA_food = pd.read_csv(os.path.join(DATA_PATH, '初赛A榜测试集', 'preliminary_a_food.csv'))
testA_submit = pd.read_csv(os.path.join(DATA_PATH, '初赛A榜测试集', 'preliminary_a_submit_sample.csv'))
# ----------------------------------------加载测试数据集------结束-------------------------------------

print(f'train_food: {train_food.shape}')
print(f'train_answer: {train_answer.shape}')
print(f'disease_feature1: {disease_feature1.shape}')
print(f'disease_feature2: {disease_feature2.shape}')
print(f'disease_feature3: {disease_feature3.shape}')

print(f'testA_food: {testA_food.shape}')
print(f'testA_submit: {testA_submit.shape}')
print()
print(f'train disease num: {train_answer.disease_id.nunique()}')
print(f'train food num: {train_answer.food_id.nunique()}')
# 排除 食物id  得到食物特征
food_feats = [item for item in train_food.columns if item not in ['food_id']]

tmp = train_food[food_feats].isna().sum()  # 统计 每个特征有多少个空的
food_feats = tmp[tmp < 346].index.tolist()  # 总特征212   75% 155
train_food = train_food[['food_id'] + food_feats]  # 重构新的food 特征

train_answer = train_answer.merge(train_food, on='food_id', how='left')

testA_food = testA_food[['food_id'] + food_feats]

testA_submit = testA_submit.merge(testA_food, on='food_id', how='left')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def tsvd(data, feats, n_components=10, name='tsvd', load=False):
    tsvd = Pipeline([
        ('std1', StandardScaler()),
        # ('tsvd', TruncatedSVD(n_components=n_components, n_iter=1000, random_state=SEED)),
        ('pca', PCA(n_components=n_components, random_state=SEED)),
        ('std2', StandardScaler()),
    ])
    tsvd.fit(data[feats])
    data_id = data['disease_id']
    deal_data = pd.DataFrame(tsvd.transform(data[feats]), columns=[f'{name}_{i}' for i in range(n_components)])
    deal_data.insert(0, 'disease_id', data['disease_id'])
    return deal_data


n_disease_tsvd = 80
disease_feature3 = tsvd(
    disease_feature3,
    [item for item in disease_feature3.columns if item not in ['disease_id']],
    n_components=n_disease_tsvd,
    name='disease3_tsvd'
)

train_answer = train_answer.merge(disease_feature3, on='disease_id', how='left')
testA_submit = testA_submit.merge(disease_feature3, on='disease_id', how='left')

disease_feature2 = tsvd(
    disease_feature2,
    [item for item in disease_feature2.columns if item not in ['disease_id']],
    n_components=n_disease_tsvd,
    name='disease2_tsvd'
)

train_answer = train_answer.merge(disease_feature2, on='disease_id', how='left')
testA_submit = testA_submit.merge(disease_feature2, on='disease_id', how='left')

disease_feature1 = tsvd(
    disease_feature1,
    [item for item in disease_feature1.columns if item not in ['disease_id']],
    n_components=n_disease_tsvd,
    name='disease1_tsvd'
)

train_answer = train_answer.merge(disease_feature1, on='disease_id', how='left')
testA_submit = testA_submit.merge(disease_feature1, on='disease_id', how='left')

train_answer.fillna(0, inplace=True)  # 填充0 代表特征不重要
testA_submit.fillna(0, inplace=True)

train_answer['disease_id_lbl'] = train_answer['disease_id'].apply(lambda x: int(x.split('_')[-1]))
testA_submit['disease_id_lbl'] = testA_submit['disease_id'].apply(lambda x: int(x.split('_')[-1]))
cat_feats += ['disease_id_lbl']
train_answer['disease_id_lbl'].nunique(), testA_submit['disease_id_lbl'].nunique()
feats = [item for item in train_answer.columns if item not in ['food_id', 'disease_id', 'related']]
print(cat_feats)
lgb_params = {
    'boosting_type': 'dart',  # dart  gbdt rf
    'objective': 'binary',  # mse mape
    'metric': ['auc', 'binary_logloss'],
    # 'max_depth': 6,
    'num_leaves': (2 ** 7) + 20,
    # 'num_leaves': 31,
    # 'min_data_in_leaf': 50,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,  # TODO
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'learning_rate': 0.2,
    'n_jobs': 6,
    'verbose': -1,
    "device_type": "cpu",
    'feature_fraction_seed': SEED,
    'bagging_seed': SEED,
    'seed': SEED,
}
train_x = train_answer[feats]
testA_x = testA_submit[feats]
train_y = train_answer['related']
group_x = train_answer['food_id']
print(train_x.shape, testA_x.shape, train_y.shape)
task_name = "lgb"
task_params = {"lgb": lgb_params}[task_name]

train_oof = np.zeros(len(train_y))
test_pred = np.zeros(len(testA_x))
fold_num = 5
importance = 0
kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
    print('-----------', fold)
    train = lgb.Dataset(
        train_x.loc[train_idx],
        train_y.loc[train_idx],
        categorical_feature=cat_feats
    )
    val = lgb.Dataset(
        train_x.loc[val_idx],
        train_y.loc[val_idx],
        categorical_feature=cat_feats
    )
    model = lgb.train(task_params, train, valid_sets=[train, val], num_boost_round=15000,
                      callbacks=[lgb.early_stopping(2000), lgb.log_evaluation(5000)])
    best_iteration = model.best_iteration
    train_oof[val_idx] += (model.predict(train_x.loc[val_idx], num_iteration=best_iteration))
    test_pred += (model.predict(testA_x, num_iteration=best_iteration)) / fold_num
    importance += model.feature_importance(importance_type='gain') / fold_num

feats_importance = pd.DataFrame()
feats_importance['name'] = feats
feats_importance['importance'] = importance
print(feats_importance.sort_values('importance', ascending=False)[:30])


def prob_post_processing(train_oof, test_pred, threshold):
    train_oof = 1 / (1 + np.exp((-train_oof + threshold) * 3))
    test_pred = 1 / (1 + np.exp((-test_pred + threshold) * 3))
    return train_oof, test_pred


def Find_Optimal_Cutoff_F1(y, prob, verbose=False):
    precision, recall, threshold = precision_recall_curve(y, prob)
    y = 2 * (precision * recall) / (precision + recall)
    Youden_index = np.argmax(y)  # 得到f1最好 情况下的阈值
    optimal_threshold = threshold[Youden_index]
    if verbose: print("optimal_threshold", optimal_threshold)
    return optimal_threshold


# -----------结果处理------------------
optimal_threshold = test_pred[test_pred.argsort()][-4655]
print('test thres', optimal_threshold)
train_oof, test_pred = prob_post_processing(train_oof, test_pred, optimal_threshold)
optimal_threshold = Find_Optimal_Cutoff_F1(train_y, train_oof, verbose=True)  # 得到最优的阈值
y_pred = (train_oof >= optimal_threshold).astype(int)
y_true = train_y

auc_value = roc_auc_score(train_y, train_oof)
f1_vlaue = f1_score(y_true, y_pred)
p_value = precision_score(y_true, y_pred)
r_value = recall_score(y_true, y_pred)

print('total_score:', (auc_value + f1_vlaue) / 2)
print("auc_score:", auc_value)
print("f1_score:", f1_vlaue)
print("precision_score(查准率):", p_value)
print("recall_score(查全率):", r_value)
score_str = f"{(auc_value + f1_vlaue) / 2:.8f}_{auc_value:.5f}_{f1_vlaue:.5f}"
train_oof = pd.DataFrame(
    {'food_id': train_answer['food_id'], 'disease_id': train_answer['disease_id'], 'related': train_answer['related'],
     'pred': train_oof})
train_oof.to_csv(f'results/lgb_oof_{score_str}.csv', index=False)

test_pred = pd.DataFrame(
    {'food_id': testA_submit['food_id'], 'disease_id': testA_submit['disease_id'], 'related_prob': test_pred})
test_pred.to_csv(f'results/lgb_pre_{score_str}.csv', index=False)
print('连接数:', (test_pred.related_prob >= 0.5).sum())
