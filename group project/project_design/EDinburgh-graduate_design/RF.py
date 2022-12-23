import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import plotly.express as px
from preprocessing import preprocessing
import sklearn.metrics as metrics

ncomp = 40
x_train, x_test, y_train, y_test = preprocessing(ncomp)

tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(x_train)

fig = px.scatter(
    projections, x=0, y=1,
    color=1, labels={'color': 'species'}
)
fig.show()

forest = RandomForestClassifier(random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)
importances = forest.feature_importances_
print("Importance: ", importances)

indices = np.argsort(importances)[::-1]
x_columns_indices = []

score_tra = forest.score(x_train, y_train)
score_test = forest.score(x_test, y_test)
print('train:', score_tra)

test_est = forest.predict(x_test)
print('test:', score_test, accuracy_score(y_test, test_est))

print("Precision:")
print(metrics.classification_report(y_test, test_est))
