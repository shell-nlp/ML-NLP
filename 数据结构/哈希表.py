hash = dict()

word_base = ("共产党","政府","毛泽东")
hash.update({k:v for v,k in enumerate(word_base)})
print(hash)