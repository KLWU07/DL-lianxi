from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# 导入数据
filename = 'pima_data.csv'  # 确保路径正确
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# 将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
# 设置留一法交叉验证
loocv = LeaveOneOut()
# 创建逻辑回归模型
model = LogisticRegression(max_iter=1000)  # 增加最大迭代次数
# 进行交叉验证
result = cross_val_score(model, X, Y, cv=loocv)
# 打印结果
print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean() * 100, result.std() * 100))