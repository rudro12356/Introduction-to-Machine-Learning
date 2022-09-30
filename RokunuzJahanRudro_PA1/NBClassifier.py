import numpy as np

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petalwidth',
         'class']
dataset = read_csv(url, names=names)

# Create Arrays for Features and Classes
array = dataset.values
X = array[:, 0:4]  # contains flower features (petal length, etc..)
y = array[:, 4]  # contains flower names

# Split Data into 1 Folds for Training and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.50, random_state=1)

model = GaussianNB()
model.fit(X_train, y_train)
predictionResult = model.predict(X_test)

# print(X_train)
print(accuracy_score(y_test, predictionResult))
print(classification_report(y_test, predictionResult))
print(confusion_matrix(y_test, predictionResult))
print(" ")
