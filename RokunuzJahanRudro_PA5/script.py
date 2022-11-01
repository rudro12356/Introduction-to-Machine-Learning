# %% [markdown]
# # Assignment 5

# %% [markdown]
# ## Part 1

# %%
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from itertools import count
from imblearn.over_sampling import ADASYN
from numpy import where
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE
import pylab as pl
from sklearn.model_selection import cross_val_predict
from numpy import mean
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from imblearn.pipeline import Pipeline
from typing import Any
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, balanced_accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
from scikitplot.metrics import plot_roc
from scikitplot.metrics import plot_precision_recall
import imblearn

# %%
# reading data from the new csv file
url = "imbalanced iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petalwidth',
         'class']
dataset = read_csv(url, names=names)
dataset.head(10)

# %%
rows, col = dataset.shape
print("Rows : %s, column : %s" % (rows, col))
type(dataset)

# %%
features_df = dataset.iloc[1:, 0:4]
features_df

# %%
target_df = dataset.iloc[1:, -1]
target_df

# %%
X = features_df.to_numpy()
y = target_df.to_numpy()
array_2 = preprocessing.LabelEncoder()
array_2.fit(y)


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, array_2.transform(y), test_size=0.5, random_state=1)

X_train_2 = X_test
y_train_2 = y_test

X_test_2 = X_train
y_test_2 = y_train

# %%
clf = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(
    100, ), random_state=1, max_iter=50000).fit(X_train, y_train)

# %%
clf_prediction = (clf.predict(X_test)).round()
clf.fit(X_train_2, y_train_2)
clfPrediction_2 = (clf.predict(X_test_2)).round()

actual = np.concatenate([y_test, y_test_2])
predicted = np.concatenate([clf_prediction, clfPrediction_2]).round()


# %%
# credit goes to  cskroonenberg
def class_balanced_score(actual, predicted):
    recall = recall_score(actual, predicted, average=None)
    precision = precision_score(actual, predicted, average=None)
    minimum = [None]*len(recall)

    # Find minimum between precision and recall
    for i in range(len(recall)):
        minimum[i] = min(recall[i], precision[i])

    # Take average of min values
    avg = sum(minimum)/len(minimum)
    return avg

# credit goes to  cskroonenberg


def balanced_accuracy(actual, predicted):
    # Calculate recall and specificity
    recall = recall_score(actual, predicted, average=None)
    CM = confusion_matrix(actual, predicted)
    FP = CM.sum(axis=0) - np.diag(CM)
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)
    S = TN/(TN+FP)

    avg = [None]*len(recall)
    # Calculate averages of recall and specificity for each class
    for i in range(len(recall)):
        avg[i] = (recall[i] + S[i])/2

    # Take average of average values
    avg = sum(avg)/len(avg)
    return avg


class_balanced_score(actual, predicted)
balanced_accuracy(actual, predicted)

# %%


def show_metrics(actual, predicted):
    print(f"Accuracy score: {accuracy_score(actual, predicted)}")
    print(f"Confusion matrix for NN model:\n " +
          str(confusion_matrix(actual, predicted)))
    print(f"F-1 score: " + str(f1_score(actual, predicted, average=None)))
    print(f"Precision: {precision_score(actual, predicted, average=None)}")
    print(f"Recall: {recall_score(actual, predicted,average=None)}")
    print(f"Class balanced score: {class_balanced_score(actual, predicted)}")
    print(f"Balanced score: {balanced_accuracy(actual, predicted)}")

    r = confusion_matrix(actual, predicted).sum()
    print(f"Sum of Matrix: {r}")


# %%
show_metrics(actual, predicted)

# %% [markdown]
# ## Part 2

# %% [markdown]
# ### Random Oversampling

# %%


# previous - summarize class distribution
print(Counter(y))
# define oversampling strategy
oversample = RandomOverSampler()

# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)

# after- summarize class distribution
print(Counter(y_over))

# %%

expected = y_over
# define pipeline
steps = [('over', RandomOverSampler()), ('model', MLPClassifier())]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = KFold(n_splits=2, random_state=5, shuffle=True)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)
y_prediction = cross_val_predict(clf, X_over, y_over, cv=2)
acc_score = accuracy_score(expected, y_prediction)
score = mean(scores)

# %%

cm = confusion_matrix(expected, y_prediction)
print(f"Accuracy score: {acc_score}")
print(f"Score: {score}")
print(f"Confusion matrix: \n{cm}")
print(f"Sum of matrix: {cm.sum()}")

pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.xlabel('Predicted')
pl.ylabel('True')

pl.colorbar()
pl.show()

# %% [markdown]
# ### SMOTE

# %%

# %%
# define oversampling strategy
oversample = SMOTE()

# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)

counter = Counter(y_over)
# after- summarize class distribution
print(counter)

# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

# %%
expected = y_over
# define pipeline
steps = [('over', SMOTE()), ('model', MLPClassifier())]
pipeline = Pipeline(steps=steps)

# %%
# evaluate pipeline
cv = KFold(n_splits=2, random_state=5, shuffle=True)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)
y_prediction = cross_val_predict(clf, X_over, y_over, cv=2)
acc_score = accuracy_score(expected, y_prediction)
score = mean(scores)

# %%

cm = confusion_matrix(expected, y_prediction)
print(f"Accuracy score: {acc_score}")
print(f"Score: {score}")
print(f"Confusion matrix: \n{cm}")
print(f"Sum of matrix: {cm.sum()}")


pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.xlabel('Predicted')
pl.ylabel('True')

pl.colorbar()
pl.show()

# %% [markdown]
# #### ADASYN

# %%

print(Counter(y))
# define oversampling strategy
oversample = ADASYN(sampling_strategy="minority", random_state=10)

# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)

counter = Counter(y_over)
# after- summarize class distribution
print(counter)

# %%
# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

# %%
expected = y_over
# define pipeline
steps = [('over', ADASYN()), ('model', MLPClassifier())]
pipeline = Pipeline(steps=steps)

# %%
# evaluate pipeline
cv = KFold(n_splits=2, random_state=5, shuffle=True)
y_prediction = cross_val_predict(clf, X_over, y_over, cv=2)
acc_score = accuracy_score(expected, y_prediction)

# %%

cm = confusion_matrix(expected, y_prediction)
print(f"Accuracy score: {acc_score}")
print(f"Score: {score}")
print(f"Confusion matrix: \n{cm}")
print(f"Sum of matrix: {cm.sum()}")


pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.xlabel('Predicted')
pl.ylabel('True')

pl.colorbar()
pl.show()

# %% [markdown]
# ## Part 3

# %% [markdown]
# ### Random Undersampling

# %%


# previous - summarize class distribution
print(f"Before sampling counter: {Counter(y)}")
# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')

# fit and apply the transform
X_under, y_under = oversample.fit_resample(X, y)

counter = Counter(y_under)
# after- summarize class distribution
print(f"After sampling counter: {counter}")

# %%
expected = y_under
# define pipeline
steps = [('over', RandomUnderSampler()), ('model', MLPClassifier())]
pipeline = Pipeline(steps=steps)

# %%
# evaluate pipeline
cv = KFold(n_splits=2, random_state=5, shuffle=True)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)
y_prediction = cross_val_predict(clf, X_under, y_under, cv=2)
acc_score = accuracy_score(expected, y_prediction)
score = mean(scores)

# %%

cm = confusion_matrix(expected, y_prediction)
print(f"Accuracy score: {acc_score}")
print(f"Score: {score}")
print(f"Confusion matrix: \n{cm}")
print(f"Sum of matrix: {cm.sum()}")


pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.xlabel('Predicted')
pl.ylabel('True')

pl.colorbar()
pl.show()

# %% [markdown]
# ### Cluster Undersampling

# %%

# previous - summarize class distribution
print(f"Before sampling counter: {Counter(y)}")

cc = ClusterCentroids(random_state=42, sampling_strategy="all")

# fit and apply the transform
X_under, y_under = cc.fit_resample(X, y)

counter = Counter(y_under)
# after- summarize class distribution
print(f"After sampling counter: {counter}")


# %%
expected = y_under
# define pipeline
steps = [('over', ClusterCentroids()), ('model', MLPClassifier())]
pipeline = Pipeline(steps=steps)

# %%
# evaluate pipeline
cv = KFold(n_splits=2, random_state=5, shuffle=True)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)
y_prediction = cross_val_predict(clf, X_under, y_under, cv=2)
acc_score = accuracy_score(expected, y_prediction)
score = mean(scores)

# %%

cm = confusion_matrix(expected, y_prediction)
print(f"Accuracy score: {acc_score}")
print(f"Score: {score}")
print(f"Confusion matrix: \n{cm}")
print(f"Sum of matrix: {cm.sum()}")


pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.xlabel('Predicted')
pl.ylabel('True')

pl.colorbar()
pl.show()

# %% [markdown]
# ### Tomek Links

# %%

# %%
tl = TomekLinks(sampling_strategy='all')

# previous - summarize class distribution
print(f"Before sampling counter: {Counter(y)}")

# fit and apply the transform
X_under, y_under = cc.fit_resample(X, y)

counter = Counter(y_under)
# after- summarize class distribution
print(f"After sampling counter: {counter}")


# %%
expected = y_under
# define pipeline
steps = [('over', TomekLinks()), ('model', MLPClassifier())]
pipeline = Pipeline(steps=steps)

# %%
# evaluate pipeline
cv = KFold(n_splits=2, random_state=5, shuffle=True)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)
y_prediction = cross_val_predict(clf, X_under, y_under, cv=2)
acc_score = accuracy_score(expected, y_prediction)
score = mean(scores)

# %%

cm = confusion_matrix(expected, y_prediction)
print(f"Accuracy score: {acc_score}")
print(f"Score: {score}")
print(f"Confusion matrix: \n{cm}")
print(f"Sum of matrix: {cm.sum()}")


pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.xlabel('Predicted')
pl.ylabel('True')

pl.colorbar()
pl.show()

# %%
