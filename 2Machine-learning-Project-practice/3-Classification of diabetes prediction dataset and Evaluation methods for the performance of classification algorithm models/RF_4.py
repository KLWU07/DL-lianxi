from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# 准备数据
X = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values

# 10折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=7)
model = RandomForestClassifier(n_estimators=100, random_state=7)

# 获取交叉验证的预测结果
y_pred = cross_val_predict(model, X, y, cv=kf)

# 计算整体准确率
print(f"10折交叉验证准确率: {accuracy_score(y, y_pred):.3f}")

# 真实值 vs 预测值散点图（添加抖动）
plt.figure(figsize=(8, 6))
plt.scatter(y + np.random.normal(0, 0.03, len(y)),  # 添加轻微抖动
            y_pred + np.random.normal(0, 0.03, len(y_pred)),
            alpha=0.5, c='blue')
plt.plot([-0.2, 1.2], [-0.2, 1.2], 'r--')  # 对角线参考线
plt.title("10折交叉验证: 真实值 vs 预测值 (添加抖动)")
plt.xlabel("真实值 (带抖动)")
plt.ylabel("预测值 (带抖动)")
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.grid(True)
plt.show()

# 混淆矩阵热力图
plt.figure(figsize=(6, 6))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测0', '预测1'],
            yticklabels=['真实0', '真实1'])
plt.title("10折交叉验证混淆矩阵")
plt.show()

# 打印分类报告
from sklearn.metrics import classification_report
print("\n分类报告:")
print(classification_report(y, y_pred))