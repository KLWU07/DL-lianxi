from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from joblib import dump, load  # 导入模型保存和加载的函数

# 1) 导入数据
categories = ['alt.atheism',
              'rec.sport.hockey',
              'comp.graphics',
              'sci.crypt',
              'comp.os.ms-windows.misc',
              'sci.electronics',
              'comp.sys.ibm.pc.hardware',
              'sci.med',
              'comp.sys.mac.hardware',
              'sci.space',
              'comp.windows.x',
              'soc.religion.christian',
              'misc.forsale',
              'talk.politics.guns',
              'rec.autos',
              'talk.politics.mideast',
              'rec.motorcycles',
              'talk.politics.misc',
              'rec.sport.baseball',
              'talk.religion.misc']
# 导入训练数据
train_path = '20news-bydate-train'
dataset_train = load_files(container_path=train_path, categories=categories)
# 导入评估数据
test_path = '20news-bydate-test'
dataset_test = load_files(container_path=test_path, categories=categories)

print("实际加载的类别数:", len(dataset_train.target_names))
print("类别名称:", dataset_train.target_names)

# 计算TF-IDF
tf_transformer = TfidfVectorizer(stop_words='english', decode_error='ignore')
X_train_counts_tf = tf_transformer.fit_transform(dataset_train.data)
# 查看数据维度
print(X_train_counts_tf.shape)

# 保存TF-IDF向量器
dump(tf_transformer, 'tfidf_vectorizer.joblib')

# 设置评估算法的基准
num_folds = 10
seed = 7
scoring = 'accuracy'

# 生成算法模型
models = {}
models['LR'] = LogisticRegression(C=15,n_jobs=-1, max_iter=1000)

# 比较算法
results = []
for key in models:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(models[key], X_train_counts_tf, dataset_train.target, cv=kfold, scoring=scoring)
    results.append(cv_results)
    print('%s : %f (%f)' % (key, cv_results.mean(), cv_results.std()))

    # 训练完整模型（不使用交叉验证）
    model = models[key]
    model.fit(X_train_counts_tf, dataset_train.target)

    # 保存模型
    model_filename = f'{key}_model.joblib'
    dump(model, model_filename)
    print(f'模型已保存为 {model_filename}')

# 保存类别名称
dump(dataset_train.target_names, 'target_names.joblib')