# 导入必要的库（保持不变）
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics

# 数据加载和预处理（保持不变）
filename = 'wine.data'
names = ['class', 'Alcohol', 'MalicAcid', 'Ash', 'AlclinityOfAsh', 'Magnesium', 'TotalPhenols', 'Flavanoids',
         'NonflayanoidPhenols', 'Proanthocyanins', 'ColorIntensiyt', 'Hue', 'OD280/OD315', 'Proline']
dataset = read_csv(filename, names=names)
dataset['class'] = dataset['class'].replace(to_replace=[1, 2, 3], value=[0, 1, 2])
array = dataset.values
X = array[:, 1:13]
y = array[:, 0]

# 数据降维和聚类（修改降维部分）
pca = PCA(n_components=2)  # 修改为2维降维
X_scale = StandardScaler().fit_transform(X)
X_reduce = pca.fit_transform(X_scale)

model = KMeans(n_clusters=3, n_init=10)
model.fit(X_reduce)
labels = model.labels_
centers = model.cluster_centers_
print(model.transform(X_reduce))

# 输出模型评估指标（保持不变）
print('%.3f   %.3f   %.3f   %.3f   %.3f    %.3f' % (
    metrics.homogeneity_score(y, labels),
    metrics.completeness_score(y, labels),
    metrics.v_measure_score(y, labels),
    metrics.adjusted_rand_score(y, labels),
    metrics.adjusted_mutual_info_score(y, labels),
    metrics.silhouette_score(X_reduce, labels)
))

# 绘制模型的分布图（修改为2D散点图）
plt.figure(figsize=(10, 6))  # 添加图形尺寸
plt.scatter(X_reduce[:, 0], X_reduce[:, 1], c=labels, cmap='tab10', s=60, alpha=0.8)  # 优化散点图配置
plt.scatter(centers[:, 0], centers[:, 1], marker='*', color='red', s=300, label='Cluster Centers')  # 添加图例标签

# 添加坐标轴标签和标题
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KMeans Clustering Result (2D PCA)')
plt.legend()  # 显示图例

plt.tight_layout()  # 优化布局
plt.show()