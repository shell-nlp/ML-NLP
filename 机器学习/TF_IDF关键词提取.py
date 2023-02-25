from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from transformers import BertTokenizer

t = BertTokenizer.from_pretrained("bert-base-chinese")

# 示例文本
text = [
    # doc 1
    "目前文本自生成技术可分为基于规动划的生成技术、基于规则的模板生成技术、基于深度学习的生成技术等。",
    # doc 2
    "NLG 技术具有极为广泛的应用价值，应用于智能问答对话系统和机器翻译系统时，可实现更为智能便捷的人机交互"]
text = [" ".join(jieba.cut(t)) for t in text]
stop_words = ["。", "、"]
# 创建TfidfVectorizer对象
vectorizer = TfidfVectorizer(stop_words=stop_words)

# 使用TfidfVectorizer进行文本特征提取
X = vectorizer.fit(text)

# 输出关键词及其权重
feature_names = vectorizer.get_feature_names_out()
for i, doc in enumerate(text):
    print("Document ", i + 1)
    tfidf_result = vectorizer.transform([doc])
    for j in tfidf_result.indices:
        print(feature_names[j], tfidf_result[0, j])
    print("\n")
# keywords = []
# for col in X.nonzero()[1]:
#     keywords.append(feature_names[col])
#
# print(keywords)
