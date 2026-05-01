from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

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
    'n_estimators': randint(10, 500),       # 树的数量范围
    'max_depth': randint(3, 30),            # 树的最大深度范围
    'min_samples_split': randint(2, 200),     # 分裂内部节点所需的最小样本数
    'min_samples_leaf': randint(1, 10),       # 叶节点所需的最小样本数
    'max_features': ['sqrt', 'log2', None]   # 寻找最佳分割时要考虑的特征数量
}

# 通过随机搜索查询最优参数
grid = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=1000,
    cv=10,  # 使用10折交叉验证
    random_state=7,
    verbose = 2,
    n_jobs=-1  # 使用所有可用的CPU核心
)

grid.fit(X, Y)

# 搜索结果
print('最高得分：%.3f' % grid.best_score_)
print('最优参数：')
best_params = grid.best_params_
for param, value in best_params.items():
    print(f"{param}: {value}")