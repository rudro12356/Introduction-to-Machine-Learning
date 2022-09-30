import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petalwidth',
         'class']
dataset = read_csv(url, names=names)
# dataset.head()
# dataset.info()

# dataset.describe()

rows, col = dataset.shape
print("Rows : %s, column : %s" % (rows, col))

# Create Arrays for Features and Classes
array = dataset.values
X = array[:, 0:4]  # contains flower features (petal length, etc..)
y = array[:, 4]  # contains flower names

# Encoding process
array_2 = preprocessing.LabelEncoder()
array_2.fit(y)

# Split Data into 2 Folds for Training and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, array_2.transform(y), test_size=0.50, random_state=1)

# New variables
X_train_2 = X_test
y_train_2 = y_test

X_test_2 = X_train
y_test_2 = y_train

# Linear regression model
LR_model = LinearRegression()
LR_model.fit(X_train, y_train)
LR_predict = (LR_model.predict(X_test)).round()
LR_accuracy = accuracy_score(y_test, LR_predict)
LR_confusion_matrix = confusion_matrix(y_test, LR_predict)

LR_model.fit(X_train_2, y_train_2)
LR_predict = (LR_model.predict(X_test_2)).round()
LR_accuracy_2 = accuracy_score(y_test_2, LR_predict)
LR_confusion_matrix_2 = confusion_matrix(y_test_2, LR_predict)

LR_accuracy_avg = (LR_accuracy + LR_accuracy_2) / 2
LR_confusion_comb = str(LR_confusion_matrix + LR_confusion_matrix_2)

# Print results
print("******************1)Linear regression**********************")
print(f"\nAccuracy score: {LR_accuracy_avg}")
print(f"\nConfusion matrix: \n{LR_confusion_comb}\n")
print("***********************************************************\n")

# Linear regression (degree 2)
LR_model_2 = LinearRegression()
poly_2 = PolynomialFeatures(degree=2, include_bias=False)

X_poly_2_train_1 = poly_2.fit_transform(X_train)
X_poly_2_test_1 = poly_2.fit_transform(X_test)
LR_model_2.fit(X_poly_2_train_1, y_train)
predLR_2 = (LR_model_2.predict(X_poly_2_test_1)).round()

for index in range(len(predLR_2)):
    if predLR_2[int(index)] < 0:
        predLR_2[int(index)] = 0
    if predLR_2[int(index)] > 2:
        predLR_2[int(index)] = 2

LRaccuracy_2 = accuracy_score(y_test, predLR_2)
LRconfusion_2 = confusion_matrix(y_test, predLR_2, labels=[0, 1, 2])

X_poly_2_train_2 = poly_2.fit_transform(X_train_2)
X_poly_2_test_2 = poly_2.fit_transform(X_test_2)
LR_model_2.fit(X_poly_2_train_2, y_train_2)

predLR_2 = (LR_model_2.predict(X_poly_2_test_2)).round()

for index in range(len(predLR_2)):
    if predLR_2[int(index)] < 0:
        predLR_2[int(index)] = 0
    if predLR_2[int(index)] > 2:
        predLR_2[int(index)] = 2

LRaccuracy_2_2 = accuracy_score(y_test_2, predLR_2)
LRconfusion_2_2 = confusion_matrix(y_test_2, predLR_2, labels=[0, 1, 2])


# Print results
print("******************2)Polynomial degree 2**********************")
print("\nAccuracy score: " + str((LRaccuracy_2+LRaccuracy_2_2)/2))
print("\nConfusion matrix: \n\n" + str(LRconfusion_2+LRconfusion_2_2))
print("*************************************************************\n")


# Linear regreesion (degree 3)
LRmodel_3 = LinearRegression()
poly_3 = PolynomialFeatures(degree=3, include_bias=False)

X_poly_3_train_1 = poly_3.fit_transform(X_train)
X_poly_3_test_1 = poly_3.fit_transform(X_test)

LRmodel_3.fit(X_poly_3_train_1, y_train)

predLR_3 = (LRmodel_3.predict(X_poly_3_test_1)).round()

for index in range(len(predLR_3)):
    if predLR_3[int(index)] < 0:
        predLR_3[int(index)] = 0
    if predLR_3[int(index)] > 2:
        predLR_3[int(index)] = 2

LR_accuracy_3 = accuracy_score(y_test, predLR_3)
LR_confusion_3 = confusion_matrix(y_test, predLR_3, labels=[0, 1, 2])

X_poly_3_train_2 = poly_3.fit_transform(X_train_2)
X_poly_3_test_2 = poly_3.fit_transform(X_test_2)

LRmodel_3.fit(X_poly_3_train_2, y_train_2)
predLR_3_2 = (LRmodel_3.predict(X_poly_3_test_2)).round()

for index in range(len(predLR_3_2)):
    if predLR_3_2[int(index)] < 0:
        predLR_3_2[int(index)] = 0
    if predLR_3_2[int(index)] > 2:
        predLR_3_2[int(index)] = 2

LR_accuracy_3_2 = accuracy_score(y_test_2, predLR_3_2)
LR_confusion_3_2 = confusion_matrix(y_test_2, predLR_3_2, labels=[0, 1, 2])

# Print the results
print("******************3)Polynomial degree 3**********************")
print("\nAccuracy score: " + str((LR_accuracy_3+LR_accuracy_3_2)/2))
print("\nConfusion matrix: \n\n" + str(LR_confusion_3+LR_confusion_3_2))
print("*************************************************************\n")


# Naive Baysian (adapted from slides)
model = GaussianNB()
model.fit(X_train, y_train)
predNB = model.predict(X_test)
accuracyNB = accuracy_score(y_test, predNB)
confusion_matrix_NB = confusion_matrix(y_test, predNB)

model.fit(X_train_2, y_train_2)
predNB_2 = model.predict(X_test_2)
accuracyNB_2 = accuracy_score(y_test_2, predNB_2)
confusion_matrix_NB_2 = confusion_matrix(y_test_2, predNB_2)

NB_accuracy_avg = (accuracyNB + accuracyNB_2) / 2
NB_confusion_comb = str(confusion_matrix_NB+confusion_matrix_NB_2)

# Print the results
print("*******************4)Naive Baysian***************************")
print(f"\nAccuracy score: {NB_accuracy_avg}")
print(f"\nConfusion matrix: \n{NB_confusion_comb}")
print("***********************************************************\n")

# kNN(KNeighborsClassifier)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predKNN = knn.predict(X_test)
accuracyKNN = accuracy_score(y_test, predKNN)
confusionKNN = confusion_matrix(y_test, predKNN)

knn.fit(X_train_2, y_train_2)
predKNN_2 = knn.predict(X_test_2)
accuracyKNN_2 = accuracy_score(y_test_2, predKNN_2)
confusionKNN_2 = confusion_matrix(y_test_2, predKNN_2)

knn_accuracy_avg = (accuracyKNN + accuracyKNN_2)/2
knn_confusion_comb = str(confusionKNN+confusionKNN_2)

print("*******************5)kNN***********************************")
print(f"\nAccuracy score: {knn_accuracy_avg}")
print(f"\nConfusion matrix: \n{knn_confusion_comb}")
print("***********************************************************\n")

# LDA
LDAmodel = LinearDiscriminantAnalysis()
LDAmodel.fit(X_train, y_train)
predLDA = LDAmodel.predict(X_test)
accuracyLDA = accuracy_score(y_test, predLDA)
confusionLDA = confusion_matrix(y_test, predLDA)

LDAmodel.fit(X_train_2, y_train_2)
predLDA_2 = LDAmodel.predict(X_test_2)
accuracyLDA_2 = accuracy_score(y_test_2, predLDA_2)
confusionLDA_2 = confusion_matrix(y_test_2, predLDA_2)

LDA_accuracy_avg = (accuracyLDA + accuracyLDA_2) / 2
LDA_confusion_comb = str(confusionLDA+confusionLDA_2)

print("*******************6)LDA***********************************")
print(f"\nAccuracy score: {LDA_accuracy_avg}")
print(f"\nConfusion matrix: \n{LDA_confusion_comb}")
print("***********************************************************\n")

# QDA
QDAmodel = QuadraticDiscriminantAnalysis()
QDAmodel.fit(X_train, y_train)

predQDA = QDAmodel.predict(X_test)
accuracyQDA = accuracy_score(y_test, predQDA)
confusionQDA = confusion_matrix(y_test, predQDA)

QDAmodel.fit(X_train_2, y_train_2)
predQDA_2 = QDAmodel.predict(X_test_2)
accuracyQDA_2 = accuracy_score(y_test_2, predQDA_2)
confusionQDA_2 = confusion_matrix(y_test_2, predQDA_2)

accuracy_QDA_avg = (accuracyQDA + accuracyQDA_2) / 2
QDA_confusion_comb = str(confusionQDA + confusionQDA_2)

print("*******************6)QDA***********************************")
print(f"\nAccuracy score: {accuracy_QDA_avg}")
print(f"\nConfusion matrix: \n{QDA_confusion_comb}")
print("***********************************************************\n")
