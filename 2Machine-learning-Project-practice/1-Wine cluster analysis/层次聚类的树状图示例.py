# 层次聚类的树状图
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# 生成示例数据（二维点，分为3个自然簇）
np.random.seed(42)
X1 = np.random.randn(10, 2) + [5, 5]   # 簇1：中心在(5,5)附近
X2 = np.random.randn(10, 2) + [0, 0]   # 簇2：中心在(0,0)附近
X3 = np.random.randn(10, 2) + [10, 0]  # 簇3：中心在(10,0)附近
X = np.vstack([X1, X2, X3])  # 合并所有点

# 计算层次聚类（使用沃德方法，最小化方差）
Z = linkage(X, method='ward', metric='euclidean')

# 绘制树状图
plt.figure(figsize=(12, 6))

# 主图：树状图
plt.subplot(1, 2, 1)
dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points (Indices)')
plt.ylabel('Distance (Ward\'s method)')

# 添加水平切割线（示例：切割为3个簇）
plt.axhline(y=20, color='r', linestyle='--', label='Cut-off for 3 clusters')
plt.legend()

# 副图：原始数据点与聚类结果
plt.subplot(1, 2, 2)
# 基于距离阈值(20)获取聚类标签
labels = fcluster(Z, t=20, criterion='distance')
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title('Clustering Result (k=3)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()