# %%
import numpy as np
import pandas as pd
from sklearn import datasets
from pandas import read_csv
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import random
import math
from sklearn.impute import SimpleImputer

plt.style.use('ggplot')

# %%
data = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petalwidth',
         'class']
dataset = read_csv(data, names=names)

# %%
rows, col = dataset.shape
print("Rows : %s, column : %s" % (rows, col))

# Create Arrays for Features and Classes
array = dataset.values
X = array[:, 0:4]  # contains flower features (petal length, etc..)
y = array[:, 4]  # contains flower names

# X

# %%
# y
target_df = pd.DataFrame(y, columns=['Column_A'])
# target_df

# %%
# Encoding process
array_2 = preprocessing.LabelEncoder()
array_2.fit(y)

# %%
# Split Data into 2 Folds for Training and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, array_2.transform(y), test_size=0.50, random_state=1)

# New variables
X_train_2 = X_test
y_train_2 = y_test

X_test_2 = X_train
y_test_2 = y_train

# %% [markdown]
# ## Part 1

# %%
# Decision Tree Classifier
dclf = DecisionTreeClassifier()
dclf.fit(X_train, y_train)
# print(tree.plot_tree(dclf))
dclfPrediction_1 = (dclf.predict(X_test)).round()
dclf.fit(X_train_2, y_train_2)
# print(tree.plot_tree(dclf))
dclfPrediction_2 = (dclf.predict(X_test_2)).round()

actual = np.concatenate([y_test, y_test_2])  # type: ignore
predicted = (np.concatenate([dclfPrediction_1, dclfPrediction_2])).round()

print("*******************DTC***********************************")
print(f"\nAccurace score: " + str(accuracy_score(actual, predicted)))
print(f"\nConfusion matrix:\n " + str(confusion_matrix(actual, predicted)))
print(f"\nFeature names:", names[0:4])

# %% [markdown]
# ## Part 2

# %%
# Trying a different way to approach the problems here on
# Loading the dataset
iris_dataset = datasets.load_iris()
df = pd.DataFrame(iris_dataset.data,
                  columns=iris_dataset.feature_names)  # type: ignore
target_column = iris_dataset.target_names  # type: ignore
target_flower_name = pd.DataFrame(target_column, columns=['Flower names'])
# target_flower_name

# %%
kfold = model_selection.KFold(n_splits=2, random_state=1, shuffle=True)

# %%
X = iris_dataset.data
y = iris_dataset.target

# %%
scaler = StandardScaler()
scaler.fit(X)

# %%
# Make an instance of the Model
pca = PCA(4)
pca = PCA(n_components=4, random_state=2020)
pca.fit(X)
#principleComponents = pca.transform(X)

# Get eigenvectors and eigenvalues
eigenvectors = pca.components_
eigenvalues = pca.explained_variance_

# %%
PoV = max(pca.explained_variance_)/sum(pca.explained_variance_)

data_scaled = pd.DataFrame(preprocessing.scale(df), columns=df.columns)
scaled_data = pca.fit_transform(data_scaled)

col = pca.fit_transform(data_scaled)
col1 = pd.DataFrame(col, columns=df.columns)

# %%
print('Features:', pd.DataFrame(pca.components_,
      columns=data_scaled.columns, index=['Z-1', 'Z-2', 'Z-3', 'Z-3']))
print("Features: ", col1.columns)

# %%
print("Variance explained by all 4 components: ",
      sum(pca.explained_variance_ratio_ * 100))
pca.explained_variance_ratio_

# plot graph
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('# of components')
plt.ylabel('Explained variance')
plt

# %%
print("Variance explained by First component =",
      np.cumsum(pca.explained_variance_ratio_ * 100)[0])
print("Variance explained by Second component =",
      np.cumsum(pca.explained_variance_ratio_ * 100)[1])
print("Variance explained by Third component =",
      np.cumsum(pca.explained_variance_ratio_ * 100)[2])
print("Variance explained by Fourth component =",
      np.cumsum(pca.explained_variance_ratio_ * 100)[3])
print(f"Eigenvalue = {eigenvalues}")
print(f"Eigenvectors = {eigenvectors}")
print('Pov=', PoV)

# %%
clf = DecisionTreeClassifier()
clf.fit(data_scaled, iris_dataset.target)

expected = iris_dataset.target.tolist()  # type: ignore
# print(expected)
scores = cross_val_score(
    clf, data_scaled, iris_dataset.target, cv=kfold)  # type: ignore
y_pred = cross_val_predict(
    clf, data_scaled, iris_dataset.target, cv=kfold).tolist()  # type: ignore
# print(y_pred)

# %%
print('Accuracy= ', accuracy_score(expected, y_pred))
print('Confusion matrix= \n', confusion_matrix(expected, y_pred))

# %% [markdown]
# ## Part 3: Simulated Annealing

# %%
data2_B = pd.DataFrame(data=np.c_[col1])

new_features_8 = pd.concat([df, col1], axis=1)
print('Shape of data =', new_features_8.shape)

print('Total 8 features:', new_features_8.head())

# %%
past_acc = 1

for x in range(100):
    rn = random.randint(2, 5)
    # print(rn)
    m = (rn/100)*32

    pn = random.random()

    # Decision Trees
    model_dt = DecisionTreeClassifier()
    model_dt.fit(new_features_8, iris_dataset.target)

    print('**********************')
    # make predictions
    expected = iris_dataset.target
    scores = cross_val_score(
        model_dt, new_features_8, iris_dataset.target, cv=kfold)
    y_pred = cross_val_predict(
        model_dt, new_features_8, iris_dataset.target, cv=kfold)

    Accuracy = accuracy_score(expected, y_pred)
    print('Accuracy= ', accuracy_score(expected, y_pred))
    conf_matrix_val = confusion_matrix(expected, y_pred)
    print('Confusion Matrix')
    print(conf_matrix_val)
    remove_n = round(m)

    if Accuracy > past_acc:
        print('Status:Improved')
        #print('Accuracy', Accuracy)
        new_features_81 = new_features_8[new_features_8.columns.to_series().sample(
            int((8+remove_n)/2))]
        print('Features', new_features_81.head())
        print('shape', new_features_81.shape)
        print(' ')

    else:

        acc_prob = math.exp(-x*(past_acc-Accuracy)/past_acc)
        print('Pr[accept]=', acc_prob)
        print('Random uniform:', pn)

        past_acc = Accuracy

        remove_n = round(m)
        if pn > acc_prob:
            new_features_81 = new_features_8[new_features_8.columns.to_series().sample(
                int((8-remove_n)/2))]
            #print('feature reject')
            print('Status:Discarded')
            print('Features', new_features_81.head())
            print('final shape', new_features_81.shape)
            print(' ')
        else:
            new_features_81 = new_features_8[new_features_8.columns.to_series().sample(
                int((8+remove_n)/2))]
            #print('feature accepted')
            print('Status:Accepted')
            print('Features', new_features_81.head())
            print('final shape', new_features_81.shape)
            print(' ')

# %% [markdown]
# ## Part 4: Genetic Algorithm

# %%
print('Part 3:Genetic algorithm ')

p4_feature = np.concatenate((iris_dataset.data, col), axis=1)

data = pd.DataFrame.from_records(p4_feature)
data.columns = ['sepal-length', 'sepal-width', 'petal-length',
                'petal-width', 'lambda1', 'lambda2', 'lambda3', 'lambda4']  # type: ignore

set1 = data.filter(['lambda1', 'sepal-length', 'sepal-width',
                   'petal-length', 'petal-width'], axis=1)
set2 = data.filter(['lambda1', 'lambda2', 'sepal-width',
                   'petal-length', 'petal-width'], axis=1)
set3 = data.filter(['lambda1', 'lambda2', 'lambda3',
                   'sepal-width', 'petal-length'], axis=1)
set4 = data.filter(['lambda1', 'lambda2', 'lambda3',
                   'lambda4', 'sepal-width'], axis=1)
set5 = data.filter(['lambda1', 'lambda2', 'lambda3',
                   'lambda4', 'sepal-length'], axis=1)
population = [set1, set2, set3, set4, set5]

# %%
for yy in range(50):

    # union
    u1 = set1.merge(set2)
    u2 = set1.merge(set3)
    u3 = set1.merge(set4)
    u4 = set1.merge(set5)
    u5 = set2.merge(set3)
    u6 = set2.merge(set4)
    u7 = set2.merge(set5)
    u8 = set3.merge(set4)
    u9 = set3.merge(set4)
    u10 = set4.merge(set5)

    # intersection
    inter1 = pd.concat([set1, set2], join='inner')
    inter2 = pd.concat([set1, set3], join='inner')
    inter3 = pd.concat([set1, set4], join='inner')
    inter4 = pd.concat([set1, set5], join='inner')
    inter5 = pd.concat([set2, set3], join='inner')
    inter6 = pd.concat([set2, set4], join='inner')
    inter7 = pd.concat([set2, set5], join='inner')
    inter8 = pd.concat([set3, set4], join='inner')
    inter9 = pd.concat([set3, set5], join='inner')
    inter10 = pd.concat([set4, set5], join='inner')

    # random select one feature
    df1 = data.sample(axis='columns')  # type: ignore
    df2 = data.sample(axis='columns')  # type: ignore
    df3 = data.sample(axis='columns')  # type: ignore
    df4 = data.sample(axis='columns')  # type: ignore

    mut = [df1, df2]
    mut2 = [df3, df4]

    for x in range(25):
        aa = random.choice(mut)
        bb = random.choice(mut2)

        population2 = [set1, set2, set3, set4, set5, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10,
                       inter1, inter2, inter3, inter4, inter5, inter6, inter7, inter8, inter9, inter10]

        # mutation add
        population2[x]['key'] = 1
        aa['key'] = 1
        bb['key'] = 1

        union_mut = pd.merge(
            population2[x], aa, on='key', how='left').drop("key", 1)  # type: ignore

        # mutation delete
        diff_3 = population2[x][~population2[x].isin(
            aa)].drop("key", 1)  # type: ignore
        diff_mut = diff_3.dropna(how='all', axis=1)

        # mutation replace
        diff_4 = population2[x][~population2[x].isin(
            aa)].drop("key", 1)  # type: ignore
        diff_mut1 = diff_4.dropna(how='all', axis=1)
        diff_mut1['key'] = 1
        rep_mut = pd.merge(diff_mut1, bb, on='key').drop(
            "key", 1)  # type: ignore

        final_set = [union_mut, diff_mut, rep_mut]
        final_random = random.choice(final_set)

        ff = final_random.sample(150)

        impu = SimpleImputer(missing_values=np.nan, strategy='mean')
        impu = impu.fit(ff)

        # Impute our data, then train
        X_train_imp = impu.transform(ff)

        model_tree = DecisionTreeClassifier()
        model_tree.fit(X_train_imp, iris_dataset.target)
        # make predictions
        expected = iris_dataset.target
        scores9 = cross_val_score(
            model_tree, X_train_imp, iris_dataset.target, cv=kfold)
        y_pred9 = cross_val_predict(
            model_tree, X_train_imp, iris_dataset.target, cv=kfold)

    print('Accuracy= ', accuracy_score(expected, y_pred))
    conf_mat = confusion_matrix(expected, y_pred)
    print('Confusion Matrix')
    print(conf_mat)
    print('Dataset shape=', ff.shape)
    #print('Features: ', dataset.feature_names)
    print('Features', ff.head())
    print(f"final shape: {X_train_imp.shape}")
    print('   ')


# %%
