from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 导入数据
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
# 将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
# 设置交叉验证参数
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
# 创建一个Pipeline，包括标准化和逻辑回归
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 数据标准化
    ('classifier', LogisticRegression(max_iter=1000))  # 逻辑回归模型
])

# 进行交叉验证
result = cross_val_score(pipeline, X, Y, cv=kfold)

# 打印结果
print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean() * 100, result.std() * 100))