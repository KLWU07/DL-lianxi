from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, auc)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# 设置中文显示（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入数据
filename = 'sonar.all-data.csv'
data = pd.read_csv(filename, header=None)

# 将目标变量转换为数值类型
data[60] = data[60].map({'R': 0, 'M': 1})

# 分离评估数据集
array = data.values
X = array[:, 0:60].astype(float)
Y = array[:, 60]
feature_names = [f'Feature_{i}' for i in range(60)]  # 创建特征名称

# 设置交叉验证参数
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

# 创建包含标准化的管道
model = Pipeline([
    ('scaler', StandardScaler()),  # 数据标准化
    ('etc', ExtraTreesClassifier(random_state=seed))  # 分类模型
])

# 训练模型用于SHAP分析（使用全部数据）
model.fit(X, Y)

# ================== 评估部分 ==================
# 1. 交叉验证获取各项指标
print("\n=== 交叉验证评估 ===")
accuracy_scores = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
precision_scores = cross_val_score(model, X, Y, cv=kfold, scoring='precision')
recall_scores = cross_val_score(model, X, Y, cv=kfold, scoring='recall')
f1_scores = cross_val_score(model, X, Y, cv=kfold, scoring='f1')
roc_auc_scores = cross_val_score(model, X, Y, cv=kfold, scoring='roc_auc')

print(f"Accuracy: {accuracy_scores.mean():.3f} (±{accuracy_scores.std():.3f})")
print(f"Precision: {precision_scores.mean():.3f} (±{precision_scores.std():.3f})")
print(f"Recall: {recall_scores.mean():.3f} (±{recall_scores.std():.3f})")
print(f"F1-score: {f1_scores.mean():.3f} (±{f1_scores.std():.3f})")
print(f"ROC-AUC: {roc_auc_scores.mean():.3f} (±{roc_auc_scores.std():.3f})")

# 2. 预测结果
y_pred = cross_val_predict(model, X, Y, cv=kfold)
y_pred_proba = cross_val_predict(model, X, Y, cv=kfold, method='predict_proba')[:, 1]

# 3. 分类报告
print("\n=== 分类报告 ===")
print(classification_report(Y, y_pred, target_names=['Rock', 'Mine']))

# 4. 混淆矩阵
cm = confusion_matrix(Y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rock', 'Mine'],
            yticklabels=['Rock', 'Mine'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 5. ROC曲线
fpr, tpr, _ = roc_curve(Y, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# ================== SHAP分析部分（改为分析Rock类别） ==================
print("\n=== SHAP特征重要性分析（Class 'Rock'） ===")

# 创建SHAP解释器
explainer = shap.TreeExplainer(model.named_steps['etc'])
shap_values = explainer.shap_values(model.named_steps['scaler'].transform(X))

# 1. SHAP摘要图（改为类别0：Rock）
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[0], model.named_steps['scaler'].transform(X),
                 feature_names=feature_names, plot_type="dot",
                 show=False)
plt.title("SHAP Feature Importance (Class 'Rock')")
plt.tight_layout()
plt.show()

# 2. SHAP条形图（平均绝对值，类别0）
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[0], model.named_steps['scaler'].transform(X),
                 feature_names=feature_names, plot_type="bar",
                 show=False)
plt.title("Mean |SHAP Value| (Class 'Rock')")
plt.tight_layout()
plt.show()

# 3. 单个样本的SHAP决策图（展示第一个样本，类别0）
sample_idx = 0
shap.decision_plot(explainer.expected_value[0],
                  shap_values[0][sample_idx],
                  features=model.named_steps['scaler'].transform(X)[sample_idx],
                  feature_names=feature_names,
                  feature_order='importance')
plt.title(f"SHAP Decision Plot for Sample {sample_idx} (Class 'Rock')")
plt.tight_layout()
plt.show()

# 4. 特征重要性对比（改为类别0）
shap.dependence_plot(0, shap_values[0], model.named_steps['scaler'].transform(X),
                    feature_names=feature_names, interaction_index=None)
plt.title("SHAP Dependence Plot for Top Feature (Class 'Rock')")
plt.show()