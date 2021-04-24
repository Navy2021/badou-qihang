import numpy as np

#1.计算余弦相似度
def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num/denom

#2.实现batchnormalization计算
def batchnormalization(x):
    return (x - np.mean(x))/np.cov(x)


X= np.array([1, 2, 3, 4])
Y = np.array([3, 2, 4, 5])

result_cs = cosine_similarity(X,Y)
result_bn= batchnormalization(X)

print(result_cs)
print(result_bn)
