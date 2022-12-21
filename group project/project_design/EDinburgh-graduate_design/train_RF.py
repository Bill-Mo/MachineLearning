import warnings

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


Date = pd.read_csv('labels.csv')

dic = {'ne': 0, 'dd': 1, 'lc': 2, 'nt': 3, 'vu': 4, 'en': 5, 'cr': 6, 'ew': 7, 'ex': 8}
Date['status'] = Date['status'].map(dic)

Date_clear = Date.drop(Date[Date['status'] == 'Remove'].index)

Date_clear = Date_clear.drop(columns=['animal_name', 'order', 'foraging_stratum'])
Date_clear = Date_clear.dropna()

Date_clear.to_csv('data_true.csv', index=None)

x, y = Date_clear.iloc[:, 1:].values, Date_clear.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

feat_labels = Date_clear.columns[1:]

features = x_train

from sklearn.decomposition import PCA
n = 6
estimator = PCA(n_components= n)   # 初始化，23维压缩至20维
# 利用训练特征决定（fit）20个正交维度的方向，并转化（transform）原训练特征
x_train = estimator.fit_transform(x_train)
print(x_train.shape)
   # 维度从23变为20
# 测试特征也按照上述的20个正交维度方向进行转化（transform）
x_test = estimator.transform(x_test)


from sklearn.manifold import TSNE
import plotly.express as px


df = Date



tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(features)

fig = px.scatter(
    projections, x=0, y=1,
    color=1, labels={'color': 'species'}
)
fig.show()


# n_estimators：森林中树的数量  n_estimators=10000
# n_jobs  整数 可选（默认=1） 适合和预测并行运行的作业数，如果为-1，则将作业数设置为核心数

forest = RandomForestClassifier(random_state=0, n_jobs=-1)

forest.fit(x_train, y_train)

# 下面对训练好的随机森林，完成重要性评估

# feature_importances_  可以调取关于特征重要程度

importances = forest.feature_importances_

print("Importance：", importances)

x_columns = Date_clear.columns[1:]

indices = np.argsort(importances)[::-1]

x_columns_indices = []

for f in range(x_train.shape[1]):
    # 对于最后需要逆序排序，我认为是做了类似决策树回溯的取值，从叶子收敛

    # 到根，根部重要程度高于叶子。

    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

    x_columns_indices.append(feat_labels[indices[f]])

print(x_columns_indices)

print(x_columns.shape[0])

print(x_columns)

print(np.arange(x_columns.shape[0]))

score_tra = forest.score(x_train, y_train)
score_test = forest.score(x_test, y_test)
print('train:', score_tra)

# 使用模型来对测试集进行预测
test_est = forest.predict(x_test)
print('test:', score_test, accuracy_score(y_test, test_est))


warnings.filterwarnings('ignore')
# 模型评估
import sklearn.metrics as metrics

print("决策树准确度:")
print(metrics.classification_report(y_test, test_est))  # 该矩阵表格其实作用不大
# print("决策树 AUC:")
# fpr_test, tpr_test, th_test = metrics.roc_curve(y_test, test_est)
# print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

# AUC 大于 0.5 是最基本的要求，可见模型精度还是比较糟糕的
# 决策树的调优技巧不多展开，将在随机森林调优部分展示

# C = confusion_matrix(y_test, test_est, labels=[0,  1,  2,  3,  4,  5, 6,  7, 8])
# print('confusion_matrix:\n',C)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix
'''
# 筛选变量（选择重要性比较高的变量）

threshold = 0.15

x_selected = x_train[:, importances > threshold]

# 可视化

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.title("红酒的数据集中各个特征的重要程度", fontsize=18)

plt.ylabel("import level", fontsize=15, rotation=90)

plt.rcParams['font.sans-serif'] = ["SimHei"]

plt.rcParams['axes.unicode_minus'] = False

for i in range(x_columns.shape[0]):
    plt.bar(i, importances[indices[i]], color='orange', align='center')

    plt.xticks(np.arange(x_columns.shape[0]), x_columns_indices, rotation=90, fontsize=15)

plt.show()
'''


from sklearn.svm import SVR  # SVM 中的回归算法


# 数据预处理，使得数据更加有效的被模型或者评估器识别
from sklearn import preprocessing
# from sklearn.externals import joblib
# 获取数据
origin_data = pd.read_csv('data_true.csv')
origin_data = origin_data.loc[:, ['status', 'altitude_breadth', 'interbirth_interval_d',
                                                  'female_maturity_d', 'weaning_age_d', 'gestation_length_d',
                                                  'density_n_km2', 'generation_length_d',
                                                  'age_first_reproduction_d', 'neonate_mass_g', 'litter_size_n', 'brain_mass_g',
                                                  'litters_per_year_y', 'max_longevity_d']]

X = origin_data.iloc[:, 1:].values
Y = origin_data.iloc[:, 0].values

'''
print(type(Y))
# print(type(Y.values))
# 总特征  按照特征的重要性排序的所有特征
all_feature = [9, 12, 6, 11, 0, 10, 5, 3, 1, 8, 4, 7, 2]
# 这里我们选取前三个特征
topN_feature = all_feature[:3]
print(topN_feature)

# 获取重要特征的数据

data_X = X[:, topN_feature]
'''

# 将每个特征值归一化到一个固定范围

# 原始数据标准化，为了加速收敛

# 最小最大规范化对原始数据进行线性变换，变换到[0,1]区间
data_X = X
#data_X = preprocessing.StandardScaler().fit_transform(X)

# 利用train_test_split 进行训练集和测试集进行分开

X_train, X_test, y_train, y_test = train_test_split(data_X, Y, test_size=0.3)

# 通过多种模型预测

model_svr1 = SVR(kernel='rbf', C=50, max_iter=10000)

# 训练

# model_svr1.fit(data_X,Y)

model_svr1.fit(X_train, y_train)

# 得分
score0 = model_svr1.score(X_train, y_train)
score1 = model_svr1.score(X_test, y_test)

print('SVR score on training set:', score0)
print('SVR score on test set:', score1)

# 使用模型来对测试集进行预测
test_est = model_svr1.predict(X_test)


#print(test_est)
