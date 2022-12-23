from sklearn import metrics
from preprocessing import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics

ncomp = 40

x_train, x_test, y_train, y_test = preprocessing(ncomp)

clf = KNeighborsClassifier(n_neighbors=6)
clf.fit(x_train, y_train)
y_pred_1 = clf.predict(x_test)

print("Precision:")
print(metrics.classification_report(y_test, y_pred_1))