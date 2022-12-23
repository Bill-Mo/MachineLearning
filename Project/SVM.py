import numpy as np
from sklearn import svm
from preprocessing import preprocessing
from sklearn.model_selection import GridSearchCV

ncomp = 40
x_train, x_test, y_train, y_test = preprocessing(ncomp)

C_test = [0.1, 1, 10]
gam_test = [0.01, 0.1, 1]

nC = len(C_test)
ngam = len(gam_test)
acc = np.zeros((nC,ngam))
X = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

param_grid = {
    'C': C_test, 
    'gamma': gam_test, 
    'kernel': ('rbf', ), 
}
svc = svm.SVC()
clf = GridSearchCV(estimator=svc, param_grid=param_grid, verbose=10, return_train_score=True)
clf.fit(X, y)
print('The best score is {}'.format(clf.best_score_))
print('The best parameters are {}'.format(clf.best_params_))