from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# 划分训练集和测试集 (70%训练，30%测试)
X = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# 随机森林参数调优
model = RandomForestClassifier(random_state=7)
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None]
}
grid = RandomizedSearchCV(model, param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=7)
grid.fit(X_train, y_train)

# 最优模型预测
best_model = grid.best_estimator_
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# 评估指标
print(f"训练集准确率: {accuracy_score(y_train, y_train_pred):.3f}")
print(f"测试集准确率: {accuracy_score(y_test, y_test_pred):.3f}")

# 真实值 vs 预测值散点图
plt.figure(figsize=(12, 5))

# 训练集散点图
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, c='blue')
plt.plot([0, 1], [0, 1], 'r--')  # 对角线参考线
plt.title("Train: true vs pre")
plt.xlabel("True")
plt.ylabel("pre")
plt.xticks([0, 1])
plt.yticks([0, 1])

# 测试集散点图
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5, c='green')
plt.plot([0, 1], [0, 1], 'r--')
plt.title("Test: true vs pre")
plt.xlabel("true")
plt.ylabel("pre")
plt.xticks([0, 1])
plt.yticks([0, 1])

plt.tight_layout()
plt.show()

# 混淆矩阵热力图
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['pre 0', 'pre 1'],
            yticklabels=['true 0', 'true 1'])
plt.title("Test Confusion Matrix")
plt.show()