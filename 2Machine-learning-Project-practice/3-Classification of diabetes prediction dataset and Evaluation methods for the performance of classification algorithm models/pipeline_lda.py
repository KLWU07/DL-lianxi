from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from matplotlib import pyplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# 定义模型
models = {}
models['LR'] = LogisticRegression(max_iter=1000)
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['SVM'] = SVC()
models['NB'] = GaussianNB()
models['RF'] = RandomForestClassifier(n_estimators=100, random_state=seed)
models['XGB'] = XGBClassifier(n_estimators=100, random_state=seed)

# 创建Pipeline，包括标准化、LDA降维和模型
pipelines = {}
for name in models:
    pipelines[name] = Pipeline([
        ('scaler', StandardScaler()),  # 数据标准化
        ('lda', LinearDiscriminantAnalysis()),  # LDA降维
        ('model', models[name])       # 模型
    ])

# 评估模型
results = []
for name in pipelines:
    result = cross_val_score(pipelines[name], X, Y, cv=kfold)
    results.append(result)
    msg = '%s: %.3f (%.3f)' % (name, result.mean(), result.std())
    print(msg)

# 图表显示
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(pipelines.keys())
pyplot.show()