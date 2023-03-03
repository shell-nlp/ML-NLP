# 导入所需模块
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集并划分训练集和测试集
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建lgb数据对象
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)
# 设置模型参数
params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": ["multi_logloss"],
    "learning_rate": 0.1,
    "num_boost_round": 20,
    "verbose": -1
}
# 100轮次
es = lgb.early_stopping(stopping_rounds=10)  # 如果验证集损失在10个iteration内没有改善，则停止训练
eval_result = {}  # 用于存储评估结果的字典
re = lgb.record_evaluation(eval_result)  # 记录评估结果到字典中
model = lgb.train(params=params, train_set=train_data, valid_sets=[test_data],
                  callbacks=[es, re], num_boost_round=60)
model.save_model('model.txt', num_iteration=model.best_iteration)
# 预测测试集并计算准确率
y_pred = model.predict(X_test, num_iteration=model.best_iteration)  # 得到每个样本属于每个类别的概率矩阵
y_pred = np.argmax(y_pred, axis=1)  # 取概率最大的类别作为预测结果
acc = accuracy_score(y_test, y_pred)  # 计算准确率
# 打印结果
print(eval_result)
print('Test accuracy:', acc)
