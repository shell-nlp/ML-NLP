from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 准备数据
s = "多模态 研究 旨在 探索 不同 模态（例如 视觉 、 听觉 、语言 ）之间 的 相互 作用，以及 如何 将 这些 模态 结合 起来 以 更好地 理解 人类 认知 和 行为。以下是 国内外 多模态 研究 的 一些 现状："

corpus = [
    s
    # "我 爱 北京 天安门",
    # "天安门 上 太阳 升",
    # "伟大 领袖 毛主席",
    # "指引 我们 向前进"
]
vectorizer = CountVectorizer()  # 将文本数据转换为文档-词频矩阵
doc_word_matrix = vectorizer.fit_transform(corpus)

# 定义LDA模型超参数
num_topics = 3
num_iterations = 10
learning_offset = 50.
learning_decay = 0.7

# 创建LDA模型对象，并拟合模型
lda = LatentDirichletAllocation(n_components=num_topics,
                                max_iter=num_iterations,
                                # learning_method='online',
                                # learning_offset=learning_offset,
                                # learning_decay=learning_decay
                                )
lda.fit(doc_word_matrix)

# 输出主题-词分布矩阵和文档的主题分布向量
print('Topic-Word Distribution:')
print(lda.components_)

print('Document-Topic Distribution:')
print(lda.transform(doc_word_matrix))
doc_topics = lda.transform(doc_word_matrix)
# 获取特征名称
feature_names = vectorizer.get_feature_names_out()

print(feature_names)
#
# 个主题中最重要的前10个单词
for topic_idx, topic in enumerate(lda.components_):

    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i] for i in topic.argsort()[:-3 - 1:-1]]))
    print()

# 输出每个文档相似度最高的1个主题
for i in range(doc_topics.shape[0]):
    top_topics = doc_topics[i].argsort()[:-3:-1]
    print("Document #{}: Top Topics = {}".format(i, top_topics))
