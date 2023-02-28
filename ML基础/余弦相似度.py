import numpy as np


# @ --> np.dot()
def cosine_similarity(x, y):
    """
    计算余弦相似度
    :param x: 向量x
    :param y: 向量y
    :return: 余弦相似度
    """
    return (x @ np.transpose(y)) / (np.linalg.norm(x) * np.linalg.norm(y))


# 示例
x = np.array([1, 2, 3]).reshape(1, 3)
y = np.array([4, 5, 6]).reshape(1, 3)
similarity = cosine_similarity(x, y)
print(similarity)
