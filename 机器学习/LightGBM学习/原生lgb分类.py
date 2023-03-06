# 导入所需模块
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 2023
# 加载数据集并划分训练集和测试集
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建lgb数据对象
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)
from sklearn.metrics import precision_score, recall_score, f1_score

# 设置模型参数
max_depth = 8
params = {
    'boosting_type': 'dart',  # gbdt  dart
    "objective": "multiclass",
    "num_class": 10,
    "metric": ["multi_error", "auc_mu", "multi_logloss"],
    "max_depth": max_depth,
    "num_leaves": 2 ** (max_depth - 1),
    # "categorical_feature": “auto”,
    "learning_rate": 0.1,
    "num_boost_round": 200,
    "verbose": 10,
    "n_jobs": 4,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,  # 用来控制bagging的频率 bagging是一种随机选择部分数据进行训练的方法，可以加速训练和防止过拟合。 通常选择3-5
    "lambda_l1": 0,
    "lambda_l2": 0,
    "device_type": "cpu",
    'feature_fraction_seed': SEED,
    'bagging_seed': SEED,
    'seed': SEED,

}
# 100轮次
es = lgb.early_stopping(stopping_rounds=50)  # 如果验证集损失在10个iteration内没有改善，则停止训练
eval_result = {}  # 用于存储评估结果的字典
re = lgb.record_evaluation(eval_result)  # 记录评估结果到字典中
model = lgb.train(params=params, train_set=train_data, valid_sets=[train_data, test_data],
                  callbacks=[es], verbose_eval=10, valid_names=["train", "test"])
model.save_model('model.txt', num_iteration=model.best_iteration)
# 预测测试集并计算准确率
y_pred = model.predict(X_test, num_iteration=model.best_iteration)  # 得到每个样本属于每个类别的概率矩阵
y_pred = np.argmax(y_pred, axis=1)  # 取概率最大的类别作为预测结果
acc = accuracy_score(y_test, y_pred)  # 计算准确率
# 打印结果
print(eval_result)
print('Test accuracy:', acc)
