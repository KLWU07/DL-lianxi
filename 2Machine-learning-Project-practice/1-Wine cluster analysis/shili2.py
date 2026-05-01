# 导入必要的库（保持不变）
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# 数据加载和预处理（保持不变）
filename = 'wine.data'
names = ['class', 'Alcohol', 'MalicAcid', 'Ash', 'AlclinityOfAsh', 'Magnesium', 'TotalPhenols', 'Flavanoids',
         'NonflayanoidPhenols', 'Proanthocyanins', 'ColorIntensiyt', 'Hue', 'OD280/OD315', 'Proline']
dataset = read_csv(filename, names=names)
dataset['class'] = dataset['class'].replace(to_replace=[1, 2, 3], value=[0, 1, 2])
array = dataset.values
X = array[:, 1:14]  # 选择所有13个特征
y = array[:, 0]

# 数据标准化（保持不变）
X_scale = StandardScaler().fit_transform(X)

# 聚类（使用全部13个特征）
model = KMeans(n_clusters=3, n_init=10)
model.fit(X_scale)
labels = model.labels_
centers = model.cluster_centers_

# 输出模型评估指标（保持不变）
print('%.3f   %.3f   %.3f   %.3f   %.3f    %.3f' % (
    metrics.homogeneity_score(y, labels),
    metrics.completeness_score(y, labels),
    metrics.v_measure_score(y, labels),
    metrics.adjusted_rand_score(y, labels),
    metrics.adjusted_mutual_info_score(y, labels),
    metrics.silhouette_score(X_scale, labels)
))