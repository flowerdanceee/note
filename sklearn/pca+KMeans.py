import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

# 创建一个100*100的随机矩阵
X = np.random.randint(low=0, high=100, size=(100, 100))

# 数据标准化
scaler = StandardScaler()
scaler.fit(X)
X_scaler = scaler.transform(X)
# print(X_scaler)

# n_components 指明了降到几维
pca_model = PCA()
pca_model.fit(X_scaler)
X_pca = pca_model.transform(X_scaler)

# 得出训练集数据的降维后的结果；也可以以测试集数据作为参数，得到降维结果。
# print(X_pca)

# 2种方法判断选几个主成分
# 方法1。pca_model.explained_variance_ratio_.cumsum()计算累计百分比，从还原数据百分比的角度选择
index1 = np.argmax(pca_model.explained_variance_ratio_.cumsum() > 0.9)
index2 = np.argmax(pca_model.explained_variance_ratio_.cumsum() > 0.95)
index3 = np.argmax(pca_model.explained_variance_ratio_.cumsum() > 0.99)
print(index1, index2, index3)
plt.plot(pca_model.explained_variance_ratio_.cumsum(), 'o-')
plt.xticks(range(len(pca_model.explained_variance_ratio_.cumsum())),
           [str(i) for i in range(1, len(pca_model.explained_variance_ratio_.cumsum()) + 1)])
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.axhline(0.95, color='k', linestyle='--', linewidth=1)
plt.title('Cumulative PVE')
plt.tight_layout()
plt.show()

# k均值聚类(适用于无监督分类。n_clusters表示数据需要分几类)
k_model = KMeans(n_clusters=3)
k_model.fit(X_pca)
# 获取数据聚类后的标签
print(k_model.labels_)
# 2D可视化
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=15, c=k_model.labels_, cmap='rainbow')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()

# 3D可视化
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=k_model.labels_, cmap='rainbow')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.tight_layout()
plt.show()

# # 方法2。留一交叉验证。选择误差最小的时候的主成分个数。
# scores_mse = []
# for k in range(1, 4):
#     model = PCA(n_components=k)
#     model.fit(X_scaler)
#     X_train_pca = model.transform(X_scaler)
#     loo = LeaveOneOut()
#     mse = -cross_val_score(LinearRegression(), X_train_pca, y_train,
#                            cv=loo, scoring='neg_mean_squared_error')
#     scores_mse.append(np.mean(mse))
#
# index = np.argmin(scores_mse)
# plt.plot(range(1, 24), scores_mse)
# plt.axvline(index + 1, color='k', linestyle='--', linewidth=1)
# plt.xlabel('Number of Components')
# plt.ylabel('Mean Squared Error')
# plt.title('Leave-one-out Cross-validation Error')
# plt.tight_layout()
