from sklearn.feature_extraction.text import TfidfVectorizer
import math
from collections import Counter, defaultdict


def compute_tf(text):
    """
    计算文本的词频
    :param text: 文本字符串
    :return: 词频字典
    """
    tf_dict = Counter(text.split())
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / len(text.split())
    return tf_dict


def compute_idf(docs):
    """
    计算逆文档频率
    :param docs: 文档列表
    :return: 逆文档频率字典
    """
    N = len(docs)
    idf_dict = defaultdict(float)
    for doc in docs:
        for word, count in doc.items():  # 文档 1 的  A : 0.37
            if count > 0:  # 如果 等于 0  就表示没出现
                idf_dict[word] += 1
    for word, count in idf_dict.items():
        idf_dict[word] = math.log(N / count)
    return idf_dict


def compute_tfidf(tf, idf):
    """
    计算TF-IDF值
    :param tf: 词频字典
    :param idf: 逆文档频率字典
    :return: TF-IDF字典
    """
    tfidf_dict = {}
    for word, tf_value in tf.items():
        tfidf_dict[word] = tf_value * idf[word]
    return tfidf_dict


def text_tfidf(docs):
    tf_list = []
    for doc in docs:
        tf_list.append(compute_tf(doc))
    idf = compute_idf(tf_list)

    tfidf_list = []
    for tf in tf_list:
        tfidf_list.append(compute_tfidf(tf, idf))
    return tfidf_list


if __name__ == '__main__':

    # 示例
    docs = [
        "i love you",
        "i love you to",
        "you love me"
    ]
    tfidf_list = text_tfidf(docs)

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(docs)

    for i in range(len(docs)):
        print(f"TF-IDF vector for sentence {i + 1}:\n {tfidf_list[i]}")
