import warnings

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
Date_clear = pd.read_csv('data_true.csv')
Date_clear = Date_clear.drop(columns=['dphy_vertebrate', 'dphy_invertebrate', 'hibernation_torpor'])

#Date_clear = Date_clear[(Date_clear > 0).all(axis=1)]
Date_clear = abs(Date_clear)

print(Date_clear.shape)

x, y = Date_clear.iloc[:, 1:].values, Date_clear.iloc[:, 0].values

# divide the data into 70% and 30%

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)




clf = MultinomialNB(alpha=1.0,fit_prior=True)
clf.fit(x_train, y_train)
y_pred_0 = clf.predict(x_train)
y_pred_1 = clf.predict(x_test)

print('Naive Bayes')
print(f'The accuracy on training set is:{accuracy_score(y_train, y_pred_0)}')
print(f'The accuracy on test set is:{accuracy_score(y_test,y_pred_1)}')
# print(f'The recall on test set is:{recall_score(y_test,y_pred_1)}, the precision is:'
#       f'{precision_score(y_test,y_pred_1)}, the f1 score is:{f1_score(y_test,y_pred_1)}')

warnings.filterwarnings('ignore')
# 模型评估
import sklearn.metrics as metrics

print("决策树准确度:")
print(metrics.classification_report(y_test, y_pred_1))  # 该矩阵表格其实作用不大


