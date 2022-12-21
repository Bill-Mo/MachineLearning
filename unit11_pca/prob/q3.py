import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
X = np.zeros(3, 5)
y = np.zeros(3)

nfold = 4

# Create a K-fold object
kf = KFold(n_splits=nfold)
kf.get_n_splits(X)

# Number of PCs to try
ncomp_test = np.arange(2,12)
num_nc = len(ncomp_test)

# Accuracy:  acc[icomp,ifold]  is test accuracy when using `ncomp = ncomp_test[icomp]` in fold `ifold`.
acc = np.zeros((num_nc,nfold))

# Loop over number of components to test
for icomp, ncomp in enumerate(ncomp_test):
    
    # Look over the folds
    for ifold, I in enumerate(kf.split(X)):
        Itr, Its = I

        # Split data into training 
        Xtr, Xts, ytr, yts = X[Itr], X[Its], y[Itr], y[Its]

        # Create a scaling object and fit the scaling on the training data
        scaling = StandardScaler()
        scaling.fit(Xtr, ytr)

        # Fit the PCA on the scaled training data
        Xtrs = scaling.transform(Xtr)
        pca = PCA(n_components=ncomp, svd_solver='randomized', whiten=True)
        pca.fit(Xtrs, ytr)
        Ztr = pca.transform(Xtrs)

        # Train a classifier on the transformed training data
        # Use a logistic regression classifier
        logreg = LogisticRegression(multi_class='auto', solver='lbfgs')

        # Transform the test data through data scaler and PCA
        Xtss = scaling.transform(Xts)
        pca.fit(Xtss, yts)
        Zts = pca.transform(Xtss)

        # Predict the labels the test data
        logreg.fit(Ztr, ytr)
        yhat = logreg.predict(Zts)

        # Measure the accuracy 
        acc[icomp, ifold] = np.mean(yhat == yts)

acc_mean = np.mean(acc, axis=1)
optimal_index = np.argmax(acc_mean)
optimal_order = ncomp_test[optimal_index]
print('optimal normal order is {}. Accurcy is {}'.format(optimal_order, acc_mean[optimal_order]))