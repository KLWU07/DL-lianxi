# 导入类库
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from time import time
from matplotlib import rcParams

# 设置中文字体和负号显示
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

# 导入数据
filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)

# 显示数据维度
print('数据维度: 行 %s，列 %s' % dataset.shape)

# 查看数据的前10行
print(dataset.head(10))

# 统计描述数据信息
print(dataset.describe())

# 分类分布情况
print(dataset.groupby('class').size())

# 箱线图
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.suptitle('各特征箱线图')  # 设置主标题，居中显示
plt.show()

# 直方图
dataset.hist()
plt.suptitle('各特征分布直方图')
plt.show()

# 散点矩阵图
scatter_matrix(dataset)
plt.suptitle('特征间散点矩阵图')
plt.show()

# 分离数据集
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=validation_size, random_state=seed)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)

# 增强的模型配置
models = {
    'LR': LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=seed),
    'LDA': LinearDiscriminantAnalysis(),
    'KNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=seed),
    'NB': GaussianNB(),
    'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=seed)
}

# 带时间记录的评估
print("\n模型训练及交叉验证结果:")
results = []
for key in models:
    start = time()
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = cross_val_score(models[key], X_train, Y_train, cv=kfold, scoring='accuracy')
    models[key].fit(X_train, Y_train)
    train_time = time() - start

    results.append(cv_results)
    print(f'{key}: 准确率 {cv_results.mean():.4f} (±{cv_results.std():.4f}) | 训练时间: {train_time:.4f}s')

# 箱线图比较算法
plt.figure(figsize=(10, 6))
plt.boxplot(results, labels=models.keys())
plt.title('算法性能比较 (10折交叉验证)')
plt.ylabel('准确率')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 全面的测试集评估
print("\n测试集性能:")
best_model = None
best_acc = 0
for name, model in models.items():
    y_pred = model.predict(X_validation)
    acc = accuracy_score(Y_validation, y_pred)
    print(f"{name} 准确率: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model = name

    if name == best_model or name == 'SVM':
        print(f"\n{name} 模型详细评估:")
        print("混淆矩阵:\n", confusion_matrix(Y_validation, y_pred))
        print("分类报告:\n", classification_report(Y_validation, y_pred))

# 特征重要性分析
plt.figure(figsize=(12, 5))

# 决策树特征重要性
plt.subplot(1, 2, 1)
importances = models['CART'].feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices], color='#1f77b4')
plt.xticks(range(X.shape[1]), np.array(names)[indices], rotation=45)
plt.title('决策树 (CART) 特征重要性')
plt.ylabel('Gini Importance')

# LDA特征系数
plt.subplot(1, 2, 2)
coef = np.mean(np.abs(models['LDA'].coef_), axis=0)
indices = np.argsort(coef)[::-1]
plt.bar(range(X.shape[1]), coef[indices], color='#ff7f0e')
plt.xticks(range(X.shape[1]), np.array(names)[indices], rotation=45)
plt.title('LDA 特征系数绝对值')
plt.ylabel('系数绝对值')

plt.tight_layout()
plt.show()

# 最佳模型的Permutation Importance
print(f"\n最佳模型 '{best_model}' 的Permutation Importance:")
result = permutation_importance(models[best_model], X_validation, Y_validation,
                                n_repeats=10, random_state=seed)
sorted_idx = result.importances_mean.argsort()[::-1]

plt.figure(figsize=(10, 5))
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(names)[sorted_idx])
plt.title(f"{best_model} Permutation Importance (测试集)")
plt.xlabel('重要性分数')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()