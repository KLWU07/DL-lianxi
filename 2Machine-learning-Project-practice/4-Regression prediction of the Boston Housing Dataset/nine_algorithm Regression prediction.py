from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 导入数据
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
         'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv(filename, names=names, delim_whitespace=True)

# 将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:13]  # 输入特征
Y = array[:, 13]    # 输出目标变量

# 设置交叉验证参数
n_splits = 10
seed = 7
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

# 定义多个回归模型
models = {
    'LinearRegression': LinearRegression(),
    'ElasticNet': ElasticNet(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'SVR': SVR(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'ExtraTreesRegressor': ExtraTreesRegressor()
}

# 定义评分标准
scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

# 定义自定义评分函数
def mean_bias_error(y_true, y_pred):
    return np.mean(y_pred - y_true)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 评估每个模型
model_names = []
rmse_scores = []
r2_scores = []

for name, model in models.items():
    print(f"Model: {name}")

    # 计算MAE
    result_mae = cross_val_score(model, X, Y, cv=kfold, scoring=scoring[0])
    print('MAE: %.3f (%.3f)' % (result_mae.mean(), result_mae.std()))

    # 计算MSE
    result_mse = cross_val_score(model, X, Y, cv=kfold, scoring=scoring[1])
    print('MSE: %.3f (%.3f)' % (result_mse.mean(), result_mse.std()))

    # 计算R^2
    result_r2 = cross_val_score(model, X, Y, cv=kfold, scoring=scoring[2])
    print('R2: %.3f (%.3f)' % (result_r2.mean(), result_r2.std()))

    # 计算MBE和RMSE
    mbe_scores = []
    rmse_scores_temp = []
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        mbe_scores.append(mean_bias_error(Y_test, Y_pred))
        rmse_scores_temp.append(root_mean_squared_error(Y_test, Y_pred))

    print('MBE: %.3f (%.3f)' % (np.mean(mbe_scores), np.std(mbe_scores)))
    print('RMSE: %.3f (%.3f)' % (np.mean(rmse_scores_temp), np.std(rmse_scores_temp)))
    print("-" * 50)

    # 保存模型名称、RMSE 和 R^2 分数
    model_names.append(name)
    rmse_scores.append(np.mean(rmse_scores_temp))
    r2_scores.append(np.mean(result_r2))

# 绘制柱状图
fig, ax1 = plt.subplots(figsize=(12, 8))

# 设置柱状图的位置
bar_width = 0.35
index = np.arange(len(model_names))

# 绘制 RMSE 柱状图
ax1.bar(index, rmse_scores, bar_width, label='RMSE', color='tab:blue')
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 创建第二个坐标轴
ax2 = ax1.twinx()

# 绘制 R^2 柱状图
ax2.bar(index + bar_width, r2_scores, bar_width, label='R^2', color='tab:red')
ax2.set_ylabel('R^2', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# 添加标题和标签
ax1.set_title('Comparison of RMSE and R^2 for Different Models')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(model_names, rotation=45, ha='right')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()