from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import matplotlib.pyplot as plt
import numpy as np

# 导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# 将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

# 算法实例化
model = RandomForestClassifier(random_state=7)

# 设置要遍历的参数
param_dist = {
    'n_estimators': randint(50, 200),       # 树的数量范围
    'max_depth': randint(3, 10),            # 树的最大深度范围
    'min_samples_split': randint(2, 10),     # 分裂内部节点所需的最小样本数
    'min_samples_leaf': randint(1, 5),       # 叶节点所需的最小样本数
    'max_features': ['sqrt', 'log2', None]   # 寻找最佳分割时要考虑的特征数量
}

# 通过随机搜索查询最优参数
grid = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,  # 使用5折交叉验证
    random_state=7,
    n_jobs=-1  # 使用所有可用的CPU核心
)

grid.fit(X, Y)

# 搜索结果
print('最高得分：%.3f' % grid.best_score_)
print('最优参数：')
best_params = grid.best_params_
for param, value in best_params.items():
    print(f"{param}: {value}")

# 使用最优模型进行特征重要性分析
best_model = grid.best_estimator_
importances = best_model.feature_importances_
feature_names = names[:-1]  # 去掉'class'列

# 按重要性排序
indices = np.argsort(importances)[::-1]

# 打印特征重要性
print("\n特征重要性排序：")
for f in range(X.shape[1]):
    print(f"{f + 1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.title("随机森林特征重要性")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.xlabel("特征")
plt.ylabel("重要性")
plt.tight_layout()
plt.show()