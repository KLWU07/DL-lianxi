1.   This article mainly introduces the 20 Newsgroups dataset and its application in text classification tasks. The 20 Newsgroups dataset contains 
approximately 20,000 newsgroup documents, which are divided into 20 newsgroups of different topics. The dataset is split into a training set 
and a test set. In the data preprocessing stage, two methods, CountVectorizer and TfidfVectorizer, were used to convert text data into numerical 
features, and finally, TF - IDF features were selected for model training and evaluation. The performance of multiple algorithms was evaluated 
through 10 - fold cross - validation, including Logistic Regression (LR), Support Vector Machine (SVM), Classification and Regression Tree (CART), 
Multinomial Naive Bayes (MNB), and K - Nearest Neighbors (KNN). Among them, SVM and LR performed relatively well. Further, grid - search parameter 
tuning was carried out on Logistic Regression, and the accuracy reached 0.9214%. Finally, the accuracy of the tuned model was verified on the test 
set, and a classification report was generated.

2.本文主要介绍了20 Newsgroups数据集及其在文本分类任务中的应用。20 Newsgroups数据集包含约20,000篇新闻组文档，分为20个不同主题的新闻组，数据集被分为训练集和测试集。
在数据预处理阶段，使用了CountVectorizer和TfidfVectorizer两种方法将文本数据转换为数值特征，最终选择了TF-IDF特征用于模型训练和评估。通过10折交叉验证评估了多种算法
的性能，包括逻辑回归（LR）、支持向量机（SVM）、分类与回归树（CART）、多项式朴素贝叶斯（MNB）和K近邻（KNN），其中SVM和LR表现较好。进一步对逻辑回归进行了网格搜索调
参准确率达到0.9214%，最终在测试集上验证了调参后的模型准确率，并生成了分类报告。

本项目更多说明，（机器学习监督学习实战六：五种算法对新闻组英文文档进行文本分类（20类），词频统计和TF-IDF 转换特征提取方法理论和对比解析）访问个人博客CSDN：https://blog.csdn.net/qq_55433305/article/details/148512784
