import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 生成随机数据
np.random.seed(42)  # 设置随机种子以确保结果可复现
X = np.random.rand(5, 13)  # 生成一个 5x13 的随机数据矩阵
print("原始数据 (5x13):")
print(X)

# 数据标准化
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
print("\n标准化后的数据 (5x13):")
print(X_scale)

# 创建 PCA 对象，指定降维到 3 维
pca = PCA(n_components=3)
X_reduce = pca.fit_transform(X_scale)  # 对标准化后的数据应用 PCA

# 查看降维后的数据
print("\n降维后的数据 (5x3):")
print(X_reduce)

# 查看主成分矩阵
print("\n主成分矩阵 (3x13):")
print(pca.components_)

# 查看方差解释率
print("\n方差解释率 (每个主成分的方差占总方差的比例):")
print(pca.explained_variance_ratio_)

# 查看累计方差解释率
print("\n累计方差解释率 (前 3 个主成分的累计方差占总方差的比例):")
print(np.cumsum(pca.explained_variance_ratio_))