from matplotlib.pyplot import xscale
import numpy as np
from sklearn import preprocessing


def max_abs_scaler(X: np.ndarray):
    """MaxAbs归一化
    公式:X_scale = X / X_max_abs,其中X_max_abs是选择列的绝对值的最大值

    Parameters
    ----------
    X : np.ndarray
         输入数据

    Returns
    -------
    np.ndarray
         返回转化后的数据
    """

    X_scaled = X/np.abs(X).max(axis=0)
    return X_scaled


if __name__ == '__main__':
    x = [[3., -1., 2., 613.],
         [2., 0., 0., 232],
         [0., 1., -1., 113],
         [1., 2., -3., 489]]
    max_abs = preprocessing.MaxAbsScaler()
    data = max_abs.fit_transform(x)
    print(data)
    data2 = max_abs_scaler(x)
    print(data2)
