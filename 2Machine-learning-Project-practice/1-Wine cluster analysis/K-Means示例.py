# K-Means 示例
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# 生成含异常值的数据
X, y = make_blobs(n_samples=300, centers=3, random_state=42)
X = np.vstack([X, [[10, 5]]])  # 添加异常值

# K-Means聚类
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='*')
plt.title('K-Means Clustering (with Outlier)')
plt.show()