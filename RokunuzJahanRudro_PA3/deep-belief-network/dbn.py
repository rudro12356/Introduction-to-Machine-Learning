#  @misc{DBNAlbert,
# title = {A Python implementation of Deep Belief Networks built upon NumPy and TensorFlow with scikit-learn compatibility},
# url = {https: // github.com/albertbup/deep-belief-network},
# author = {albertbup},
# year = {2017}}


from dbn import SupervisedDBNClassification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np

np.random.seed(1337)  # for reproducibility


# Loading dataset
digits = load_digits()
X, Y = digits.data, digits.target

# print(X.shape)
# print(Y.shape)

# Data scaling
X = (X / 16).astype(np.float32)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# print(len(X_train))
# print(len(X_test))

# print(len(Y_train))
# print(len(Y_test))

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
