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
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petalwidth',
         'class']
dataset = read_csv(url, names=names)


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

# Decision Tree Classifier
dclf = tree.DecisionTreeClassifier()
dclf.fit(X_train, y_train)
# print(tree.plot_tree(dclf))
dclfPrediction_1 = (dclf.predict(X_test)).round()
dclf.fit(X_train_2, y_train_2)
# print(tree.plot_tree(dclf))
dclfPrediction_2 = (dclf.predict(X_test_2)).round()

actual = np.concatenate([y_test, y_test_2])
predicted = (np.concatenate([dclfPrediction_1, dclfPrediction_2])).round()

print("*******************9)DTC***********************************")
print(f"\nAccurace score: " + str(accuracy_score(actual, predicted)))
print(f"\nConfusion matrix:\n " + str(confusion_matrix(actual, predicted)))
