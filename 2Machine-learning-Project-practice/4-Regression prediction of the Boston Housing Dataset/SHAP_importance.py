from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import shap

# 导入数据
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
         'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv(filename, names=names, delim_whitespace=True)

# 将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:13]  # 输入特征
Y = array[:, 13]    # 输出目标变量

# 定义多个回归模型
models = {
    'GradientBoostingRegressor': GradientBoostingRegressor(
        learning_rate=0.12829684257109286,
        max_depth=5,
        max_features='log2',
        min_samples_leaf=3,
        min_samples_split=5,
        n_estimators=122,
        random_state=42
    )
}

# 定义自定义评分函数
def mean_bias_error(y_true, y_pred):
    return np.mean(y_pred - y_true)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 评估每个模型
for name, model in models.items():
    print(f"Model: {name}")

    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 训练模型
    model.fit(X_train, Y_train)
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    # 计算性能指标
    mae = mean_absolute_error(Y_test, Y_test_pred)
    mse = mean_squared_error(Y_test, Y_test_pred)
    rmse = root_mean_squared_error(Y_test, Y_test_pred)
    r2 = r2_score(Y_test, Y_test_pred)
    mbe = mean_bias_error(Y_test, Y_test_pred)

    print(f'MAE: {mae:.3f}')
    print(f'MSE: {mse:.3f}')
    print(f'RMSE: {rmse:.3f}')
    print(f'MBE: {mbe:.3f}')
    print(f'R2: {r2:.3f}')
    print("-" * 50)

    # 绘制训练集的折线图
    plt.figure(figsize=(12, 6))
    plt.plot(Y_train, label='Train True', color='blue')
    plt.plot(Y_train_pred, label='Train Predicted', color='orange')
    plt.xlabel('Sample Index')
    plt.ylabel('MEDV')
    plt.title(f'{name} - Train True vs Predicted Values')
    plt.legend()
    plt.show()

    # 绘制测试集的折线图
    plt.figure(figsize=(12, 6))
    plt.plot(Y_test, label='Test True', color='blue')
    plt.plot(Y_test_pred, label='Test Predicted', color='orange')
    plt.xlabel('Sample Index')
    plt.ylabel('MEDV')
    plt.title(f'{name} - Test True vs Predicted Values')
    plt.legend()
    plt.show()

    # 绘制散点图
    plt.figure(figsize=(12, 6))
    plt.scatter(Y_train, Y_train_pred, label='Train', color='blue')
    plt.scatter(Y_test, Y_test_pred, label='Test', color='orange')
    plt.plot([min(Y), max(Y)], [min(Y), max(Y)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{name} - True vs Predicted Values')
    plt.legend()
    plt.show()

    # 计算特征重要性
    feature_importances = model.feature_importances_
    feature_names = names[:-1]  # 排除目标变量 'MEDV'

    # 绘制特征重要性条形图
    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, feature_importances, color='blue')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'{name} - Feature Importances')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # 可视化 SHAP 值的摘要图
    shap.summary_plot(shap_values, X_train, feature_names=feature_names)