import numpy as np
from sklearn import preprocessing


if __name__ == '__main__':
    x = [[3., -1., 2., 613.],
         [2., 0., 0., 232],
         [0., 1., -1., 113],
         [1., 2., -3., 489]]
    max_abs_scaler = preprocessing.MaxAbsScaler()
    data = max_abs_scaler.fit_transform(x)
    print(data)
