import numpy as np
# 1.concatenate
# concatenate实现数组拼接
a = np.array([[1, 2], [-3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)
# array([[1, 2],
#        [3, 4],
#        [5, 6]])
np.concatenate((a, b.T), axis=1)
# array([[1, 2, 5],
#        [3, 4, 6]])
np.abs(a)
# 对a中的元素取绝对值
np.dot(a,b)
# 矩阵相乘
np.linalg.norm(x,ord=2,axis=None,keepdims=False)
# 求模操作：ord为范数，axis为1对行向量，为0对列向量，keepdims为是否保留维数特性
np.hstack()
np.vstack()
np.transpose()

# 随机取一个矩阵数组的某几行/列
array = np.array([0, 0])
for i in range(10):
    array = np.vstack((array, [i + 1, i + 1]))
print(array)
# [[ 0  0]
#  [ 1  1]
#  [ 2  2]
#  [ 3  3]
#  [ 4  4]
#  [ 5  5]
#  [ 6  6]
#  [ 7  7]
#  [ 8  8]
#  [ 9  9]
#  [10 10]]
rand_arr = np.arange(array.shape[0])
np.random.shuffle(rand_arr)
print(array[rand_arr[0:5]])
# [[9 9]
#  [4 4]
#  [1 1]
#  [5 5]
#  [8 8]]
np.random.shuffle(rand_arr)
print(array[rand_arr[0:5]])
# [[10 10]
#  [ 3  3]
#  [ 4  4]
#  [ 8  8]
#  [ 5  5]]