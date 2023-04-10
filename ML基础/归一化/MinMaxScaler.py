import numpy as np


def min_max_scaler(X: np.ndarray, feature_range=(0, 1), eps=1e-8):
    """MinMax归一化
    公式: Z = (X-X_min)/(X_max-X_min)
    X_scale = Z*(max-min)+min   其中min,max 是feature_range的最小值和最大值
    Parameters
    ----------
    X : np.ndarray / list
        数据类型可以是(n_sample,) or (n_sample, feature_num)
    feature_range : tuple, optional
        归一化到feature_range范围, by default (0, 1)
    eps : folat, optional
        eps是一个极小值，作用在公式分母上，保证分母不为零, by default 1e-8

    Returns
    -------
    np.ndarray
        归一化后的数据
    """
    if isinstance(X, list):
        X = np.array(X)
    X_std = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0)+eps)
    X_scaled = X_std*(feature_range[1]-feature_range[0])+feature_range[0]
    return X_scaled


if __name__ == '__main__':
    x = [[3., -1., 2., 613.],
              [2., 0., 0., 232],
              [0., 1., -1., 113],
              [1., 2., -3., 489]]
    data = min_max_scaler(x)
    print(data)
    
