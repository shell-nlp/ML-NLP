import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np


def pandas2onehot(df: pd.DataFrame, col_names: list):
    """将pandas.DataFrame类型的某一列或者多列转换为onehot编码

   Parameters
   ----------
    df : pd.DataFrame
      DataFrame类型的数据
    col_names : list
      列名，是一个列表类型

    Returns
    -------
    pd.DataFrame
       返回经过oneHot编码且拼接后的内容
    """
    temp = pd.DataFrame()
    for col in col_names:
        # 使用get_dummies()对“color”列进行one-hot编码
        one_hot_encoded = pd.get_dummies(df[col], prefix=col)
        temp = pd.concat([temp, one_hot_encoded], axis=1)
    # 将编码后的结果与原始数据合并
    df = pd.concat([df, temp], axis=1)

    return df


def missing_fill(df: pd.DataFrame, strategy="constant", fill_value=None):
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    # 对数据进行fit和transform操作
    data_imputed = imputer.fit_transform(df)
    return data_imputed


if __name__ == '__main__':
    # pandas2onehot
    # 'red', 'blue', 'green', 'red', 'yellow'
    df = pd.DataFrame({'color_A': [1, 2, 6, 5, 8], "color_B": [
                      'red', 'blue', 'green', 'red', 'yellow']})
    df = pandas2onehot(df, ["color_B", "color_A"])
    print(df)

    # missing_fill
    data = np.array([[1, 2, np.nan], [3, np.nan, 5],
                    [np.nan, 6, 7], [8, 9, 10]])
    data = pd.DataFrame(data=data)
    data = missing_fill(data)

    print(data)
