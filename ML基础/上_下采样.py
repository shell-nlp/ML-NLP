from sklearn.datasets import make_classification
from collections import Counter

X, y = make_classification(
    n_samples=5000,  # 样本数
    n_features=2,  # 特征数
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=3,  # 三分类
    n_clusters_per_class=1,
    weights=[0.01, 0.05, 0.94],  # 三个分类的数据比例，数据不均衡
    class_sep=0.8,
    random_state=0,
)
print(Counter(y))
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# 上采样/过采样: 增加正样本
ros = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)

sort_y = sorted(Counter(y_resampled).items())
print(sort_y)
