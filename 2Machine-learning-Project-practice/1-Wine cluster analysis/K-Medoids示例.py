# K-Medoids 示例
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# 生成相同数据
X, y = make_blobs(n_samples=300, centers=3, random_state=42)
X = np.vstack([X, [[10, 5]]])  # 添加异常值

# K-Medoids聚类
kmedoids = KMedoids(n_clusters=3, random_state=42, method='pam')
labels = kmedoids.fit_predict(X)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(X[kmedoids.medoid_indices_, 0], X[kmedoids.medoid_indices_, 1], s=200, c='red', marker='*')
plt.title('K-Medoids Clustering (with Outlier)')
plt.show()