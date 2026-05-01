from joblib import load

# 加载组件
tf_transformer = load('tfidf_vectorizer.joblib')  # 加载TF-IDF向量化器
model = load('LR_model.joblib')                   # 加载训练好的模型
target_names = load('target_names.joblib')        # 加载类别名称

# 1. 读取 `txt` 文件内容
file_path = 'car_1.txt'  # 确保文件路径正确
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# 2. 向量化文本
X_new = tf_transformer.transform([text])  # 注意：输入必须是列表形式（即使单样本）

print(X_new.shape)
print(X_new.toarray())
# 3. 预测类别
predicted = model.predict(X_new)
predicted_class = target_names[predicted[0]]  # 获取类别名称

# 4. 输出结果
print(f"文件 '{file_path}' 的预测类别是: {predicted_class}")