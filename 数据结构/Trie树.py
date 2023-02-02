# 1.Trie树的概念
# Trie树是数据结构比较简单的一种。Trie 树的基本用法是高效的存储和查找字符串集合的数据结构。
# Trie树也叫做字典树，它是一个树形结构。是一种专门处理字符串匹配的数据结构，用来解决在一组字符串集合中快速查找某个字符串。
# Trie树本质，利用字符串之间的公共前缀，将重复的前缀合并在一起。

# 2.Trie树的三个性质：
#   根节点不包含字符，除根节点外每一个节点都只包含一个字符
#    从根节点到某一节点，路径上经过的字符连接起来，为该节点对应的字符串
#    每个节点的所有子节点包含的字符都不相同
class Trie(object):
    """
    实现1：通过python自带的字典结构
    具有如下基本功能：
    （1）根据一组words进行TrieTree的构建
    （2）添加某个word
    （3）查询某个word
    （4）删除某个word
    """
    def __init__(self):
        self.trie = {}
        self.count = 0

    def __repr__(self):
        return str(self.trie)

    def buildTree(self, wordList):
        for word in wordList:
            t = self.trie             # 指向各节点的指针，初始化为root节点
            for w in word:
                if w not in t:
                    t[w] = {'count': 0}
                t[w]['count'] += 1
                t = t[w]

            self.count += 1
            t['end'] = 1

    def add(self, word):
        t = self.trie
        for w in word:
            if w not in t:
                t[w] = {'count': 0}
            t[w]['count'] += 1
            t = t[w]

        self.count += 1
        t['end'] = 1

    def delete(self, word):
        # 仅仅改变end和count属性，字符串仍存在于存储中
        # 先确定是否存在，若存在沿着的每条路径的count都需要-1
        if not self.search(word):
            return False

        t = self.trie
        for w in word:
            t = t[w]
            t['count'] -= 1

        self.count -= 1
        t['end'] = 0


    def search(self, word):
        t = self.trie
        for w in word:
            if w not in t:
                return False
            t = t[w]
        if t.get('end') == 1:
            return True
        return False

    def prefix_count(self, prefix):
        t = self.trie
        for w in prefix:
            if w not in t:
                return -1
            t = t[w]
        return t['count']

if __name__ == '__main__':
    trie = Trie()
    trie.buildTree(["abdc","abckl"])