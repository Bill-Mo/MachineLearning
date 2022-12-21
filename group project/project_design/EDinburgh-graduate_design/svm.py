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
import os

Date_clear = pd.read_csv('data_true.csv')
# Date_clear = Date_clear.drop(columns=['dphy_vertebrate', 'dphy_invertebrate', 'hibernation_torpor'])

# Date_clear = Date_clear[(Date_clear > 0).all(axis=1)]
# Date_clear = abs(Date_clear)

dd_data = np.where(Date_clear['status'] == 1.0)[0]
Date_clear = Date_clear.drop(Date_clear.index[dd_data])

for i in np.unique(Date_clear['status']): 
   data_idx = np.where(Date_clear['status'] == i)[0]
   # print(i, data_idx)
   if len(data_idx) <= 2: 
      Date_clear = Date_clear.drop(Date_clear.index[data_idx])

x, y = Date_clear.iloc[:, 1:].values, Date_clear.iloc[:, 0].values
# divide the data into 70% and 30%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)


from sklearn.decomposition import PCA

# estimator = PCA(n_components=20)   # 初始化，64维压缩至20维
# # 利用训练特征决定（fit）20个正交维度的方向，并转化（transform）原训练特征
# X_train = estimator.fit_transform(X_train)
# print(X_train.shape)
#    # 维度从23变为20
# # 测试特征也按照上述的20个正交维度方向进行转化（transform）
# X_test = estimator.transform(X_test)
# print(X_train[0, :])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
pca = PCA(n_components=10)
pca.fit(X_train, y_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# 训练模型
model = OneVsRestClassifier(svm.SVC(C=2.8))
# print("[INFO] Successfully initialize a new model !")
# print("[INFO] Training the model…… ")
clt = model.fit(X_train, y_train)
# print("[INFO] Model training completed !")
# 保存训练好的模型，下次使用时直接加载就可以了
path = os.getcwd()
joblib.dump(clt, path + "conv_19_80%.pkl")
# print("[INFO] Model has been saved !")

y_test_pred = clt.predict(X_test)
ov_acc = metrics.accuracy_score(y_test_pred, y_test)
print("overall accuracy: %f" % (ov_acc))
print("===========================================")
acc_for_each_class = metrics.precision_score(y_test, y_test_pred, average=None, zero_division=0)
print("acc_for_each_class:\n", acc_for_each_class)
print("===========================================")
avg_acc = np.mean(acc_for_each_class)
# print("average accuracy:%f" % (avg_acc))



# print("决策树准确度:")
# print(metrics.classification_report(y_test, y_test_pred, zero_division=0))  # 该矩阵表格其实作用不大