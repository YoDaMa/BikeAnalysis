import pylab as pl 
import numpy as np 
from sklearn.datasets import fetch_lfw_people 

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)


X = lfw_people.data 
y = lfw_people.target 
print(y)
names = lfw_people.target_names 

n_samples, n_features = X.shape # X is a matrix, so the x y are rows columns. 
_, h, w = lfw_people.images.shape  # Seems to be 3D so height and width
n_classes = len(names)

print("n_samples: {}".format(n_samples))
print("n_features: {}".format(n_features))
print("n_classes: {}".format(n_classes))


def plot_gallery(images, titles, h, w, n_row=3, n_col=6):
	"""Helper fn to plt a gallery of portraits"""
	pl.figure(figsize=(1.7 * n_col, 2.3 * n_row))
	pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
	for i in range(n_row * n_col):
		pl.subplot(n_row, n_col, i + 1)
		pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
		pl.title(titles[i],size=12)
		pl.xticks(())
		pl.yticks(())


plot_gallery(X, names[y], h, w)


pl.figure(figsize=(14,3))
y_unique = np.unique(y) 
counts = [(y==i).sum() for i in y_unique]
"""
for each value in y_unique as i:
	if y==i, then 1, otherwise 0.
	sum all values of 1 and zero produced.
result is array of counts of each unique value.

Note: The sum function sums along the rows
"""
pl.xticks(y_unique, names[y_unique])
locs, labels = pl.xticks()
pl.setp(labels,rotation=45,size=20)
_ = pl.bar(y_unique,counts)

from sklearn.cross_validation import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


"""
Unsupervised Feature Extraction with Randomized PCA (Principal Component Analysis)

Linear Dimensionality Reduction using approximated Singular Value Decomposition
of the data and keeping only the most significant singular vectors to project the
data to a lower dimensional space.
"""
from sklearn.decomposition import RandomizedPCA 

n_components = 150 

print("Extracting the top {} eigenfaces from {} faces".format(n_components, X_train.shape[0]))

pca = RandomizedPCA(n_components=n_components, whiten=True)
pca.fit(X_train)


eigenfaces = pca.components_.reshape((n_components, h, w))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)


# Projecting the input data on the eigenfaces orthonormal basis
X_train_pca = pca.transform(X_train)

from sklearn.svm import SVC 

svm = SVC(kernel='rbf', class_weight='auto')

svm

from sklearn.cross_validation import StratifiedShuffleSplit 
from sklearn.cross_validation import cross_val_score 

cv = StratifiedShuffleSplit(y_train, test_size=0.20, n_iter=3)

svm_cv_scores = cross_val_score(svm, X_train_pca, y_train, scoring='f1', n_jobs=2)
svm_cv_scores


svm_cv_scores.mean(), svm_cv_scores.std()

from sklearn.grid_search import GridSearchCV 

param_grid = {
	'C': [1e3, 5e3, 1e4, 1e5],
	'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}

clf = GridSearchCV(svm, param_grid, scoring='f1', cv=cv, n_jobs=2)

clf = clf.fit(X_train_pca, y_train)

print("Best estimator found by randomized hyper parameter search:")
print(clf.best_params_) 
print("Best parameters validation score: {:.3f}".format(clf.best_score_))


X_test_pca = pca.transforom(X_test)
y_pred = clf.predict(X_test_pca)

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)
pl.show()