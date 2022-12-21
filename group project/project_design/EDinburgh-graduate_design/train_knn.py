import warnings
import joblib
import numpy as np
from pandas.core.common import random_state
from sklearn import svm, metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix

Date_clear = pd.read_csv('data_true.csv')
Date_clear = Date_clear.drop(columns=['dphy_vertebrate', 'dphy_invertebrate', 'hibernation_torpor'])

# Date_clear = Date_clear[(Date_clear > 0).all(axis=1)]
Date_clear = abs(Date_clear)

print(Date_clear.shape)

x, y = Date_clear.iloc[:, 1:].values, Date_clear.iloc[:, 0].values

# divide the data into 70% and 30%

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

warnings.filterwarnings('ignore')


from sklearn.decomposition import PCA
estimator = PCA(n_components=8)   # 初始化，64维压缩至20维
# 利用训练特征决定（fit）20个正交维度的方向，并转化（transform）原训练特征
X_train = estimator.fit_transform(X_train)
print(X_train.shape)
   # 维度从23变为20
# 测试特征也按照上述的20个正交维度方向进行转化（transform）
X_test = estimator.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train, y_train)

y_pred_1 = clf.predict(X_test)
# 模型评估
import sklearn.metrics as metrics

print("决策树准确度:")
print(metrics.classification_report(y_test, y_pred_1))  # 该矩阵表格其实作用不大


