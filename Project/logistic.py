from preprocessing import preprocessing
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

ncomp_test = [10, 20, 25, 30, 40, 57]
acc_test = np.zeros(len(ncomp_test))

col = 6
C = np.zeros((col, col, len(ncomp_test)))

for i, ncomp in enumerate(ncomp_test): 
    x_train, x_test, y_train, y_test = preprocessing(ncomp)

    logreg = LogisticRegression(C=1, solver='liblinear').fit(x_train, y_train)

    yhat = logreg.predict(x_test)
    acc = np.mean(yhat == y_test)
    acc_test[i] = acc

    c = confusion_matrix(y_test, yhat)
    c = c / c.sum()
    C[:, :, i] = c

C = np.sum(C, axis=-1) / C.shape[-1]
mean_acc = np.mean(acc_test)
se_acc = np.std(acc_test) / np.sqrt(len(ncomp_test) - 1)

print('Confusion matrix: ')
print(np.array_str(C, precision=4, suppress_small=True))
print('Mean: \t{}\nSE: \t{}'.format(mean_acc, se_acc))
opt_idx = np.argmax(acc_test)
best_acc = acc_test[opt_idx]
opt_ncomp = ncomp_test[opt_idx]
print('Best ncomp: {}'.format(opt_ncomp))
print('Highest accuracy: {}'.format(best_acc))

