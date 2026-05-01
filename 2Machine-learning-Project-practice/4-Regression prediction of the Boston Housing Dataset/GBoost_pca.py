from pandas import read_csv
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
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

# 定义回归模型
models = {
    'GradientBoostingRegressor': GradientBoostingRegressor()
}

# 定义评分标准
scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

# 定义自定义评分函数
def mean_bias_error(y_true, y_pred):
    return np.mean(y_pred - y_true)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 应用 PCA 降维
pca = PCA(n_components=2)  # 降维到 2 个主成分
X_pca = pca.fit_transform(X)

# 评估每个模型
for name, model in models.items():
    print(f"Model: {name}")

    # 计算MAE
    result_mae = cross_val_score(model, X_pca, Y, cv=kfold, scoring=scoring[0])
    print('MAE: %.3f (%.3f)' % (result_mae.mean(), result_mae.std()))

    # 计算MSE
    result_mse = cross_val_score(model, X_pca, Y, cv=kfold, scoring=scoring[1])
    print('MSE: %.3f (%.3f)' % (result_mse.mean(), result_mse.std()))

    # 计算MBE和RMSE
    mbe_scores = []
    rmse_scores = []
    for train_index, test_index in kfold.split(X_pca):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        mbe_scores.append(mean_bias_error(Y_test, Y_pred))
        rmse_scores.append(root_mean_squared_error(Y_test, Y_pred))

    print('MBE: %.3f (%.3f)' % (np.mean(mbe_scores), np.std(mbe_scores)))
    print('RMSE: %.3f (%.3f)' % (np.mean(rmse_scores), np.std(rmse_scores)))

    # 计算R^2
    result_r2 = cross_val_score(model, X_pca, Y, cv=kfold, scoring=scoring[2])
    print('R2: %.3f (%.3f)' % (result_r2.mean(), result_r2.std()))
    print("-" * 50)

    # 可视化训练集和测试集的真实值与预测值
    X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=seed)
    model.fit(X_train, Y_train)
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    # 绘制训练集的折线图
    plt.figure(figsize=(12, 6))
    plt.plot(Y_train, label='Train True', color='blue')
    plt.plot(Y_train_pred, label='Train Predicted', color='orange')
    plt.xlabel('Sample Index')
    plt.ylabel('MEDV')
    plt.title(f'{name} - Train True vs Predicted Values (PCA)')
    plt.legend()
    plt.show()

    # 绘制测试集的折线图
    plt.figure(figsize=(12, 6))
    plt.plot(Y_test, label='Test True', color='blue')
    plt.plot(Y_test_pred, label='Test Predicted', color='orange')
    plt.xlabel('Sample Index')
    plt.ylabel('MEDV')
    plt.title(f'{name} - Test True vs Predicted Values (PCA)')
    plt.legend()
    plt.show()

    # 绘制散点图
    plt.figure(figsize=(12, 6))
    plt.scatter(Y_train, Y_train_pred, label='Train', color='blue')
    plt.scatter(Y_test, Y_test_pred, label='Test', color='orange')
    plt.plot([min(Y), max(Y)], [min(Y), max(Y)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{name} - True vs Predicted Values (PCA)')
    plt.legend()
    plt.show()