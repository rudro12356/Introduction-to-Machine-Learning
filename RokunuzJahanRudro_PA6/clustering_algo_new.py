#!/usr/bin/env python
# coding: utf-8

# In[183]:


#import libraries
from sklearn.cluster import KMeans
from pandas import read_csv
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from kneed import DataGenerator, KneeLocator
from matplotlib import pyplot as plt
from sklearn import datasets
import itertools
from scipy import linalg
import matplotlib as mpl

from sklearn import mixture


# ## Part 1: k-Means Clustering

# In[3]:


iris = datasets.load_iris()
features = iris.data
target = iris.target


# In[4]:


from sklearn import preprocessing
Standardisation = preprocessing.StandardScaler()

# Scaled feature
x_after_Standardisation = Standardisation.fit_transform(features)


# In[39]:


# function to run k-means for various values of k
# https://www.codingninjas.com/codestudio/library/applying-k-means-on-iris-dataset
def run_kmeans(k):
    nclusters = 3 # this is the k in kmeans
    seed = 0

    km = KMeans(n_clusters=nclusters, random_state=seed)
    km.fit(x_after_Standardisation)

    # predict the cluster for each data point
    y_cluster_kmeans = km.predict(x_after_Standardisation)
    y_cluster_kmeans
    kmeans = KMeans(n_clusters=k,init='k-means++', max_iter=300, n_init=3, random_state=0)
    kmeans.fit(x_after_Standardisation)

    cost.append(kmeans.inertia_)


# In[40]:


# reference: https://www.geeksforgeeks.org/ml-determine-the-optimal-value-of-k-in-k-means-clustering/
# run loop for k values
cost = []
# starting index as 2 since it makes no sense to have k =1 as it's one single big cluster
for k in range(2,21):
    run_kmeans(k)

# plot the cost against K values
plt.plot(range(2, 21), cost, color ='g', linewidth ='3')
plt.xlabel("Value of K")
plt.ylabel("Squared Error (Cost)")
plt.show() # clear the plot


# In[41]:


# https://www.kaggle.com/code/kevinarvai/knee-elbow-point-detection
from kneed import KneeLocator, DataGenerator


# In[43]:


cost = []
for k in range(2,20):
    run_kmeans(k)

kl = KneeLocator(range(2, 20), cost, curve="convex", direction="decreasing")
print('From the algorithm elbow_k=',kl.elbow)


# In[44]:


plt.style.use("fivethirtyeight")
plt.plot(range(2, 20), cost)
plt.xticks(range(2, 20))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Reconstruction error vs. k using k-Means Clustering") 
plt.show()


# In[45]:


elbow_k=kl.elbow
k=elbow_k


# In[49]:


#####k=Elbow
print('\nClassifying the dataset using k=elbow_k')
kmeans = KMeans(init="random",n_clusters=elbow_k)
kmeans.fit(x_after_Standardisation)
y_kmeans = kmeans.predict(x_after_Standardisation)
print('Cannot calculate Accuracy Score because the number of classes is not the same as the number of clusters')
conf_mat = confusion_matrix(target, y_kmeans)
print('Confusion Matrix')
print(conf_mat)


# In[62]:


#k=3
print('\nClassifying the dataset using k=3')
kmeans = KMeans(init="k-means++",n_clusters=3)
kmeans.fit(x_after_Standardisation)
y_kmeans = kmeans.predict(x_after_Standardisation)
acc=accuracy_score(target, kmeans.labels_)
print("Accuracy score is", acc)
conf_mat = confusion_matrix(target, y_kmeans)
print('Confusion Matrix')
print(conf_mat)


# ## Part 2 : GMM

# In[257]:


from sklearn.mixture import GaussianMixture

aic = []

for k in range(2, 20):
    gmm = GaussianMixture(n_components = 3,init_params='k-means++', random_state=None).fit(x_after_Standardisation)
    gmm.aic(x_after_Standardisation)
    aic.append(gmm.aic(x_after_Standardisation))

aic


# In[258]:


kneed = KneeLocator(range(2, 20), aic, curve="concave", direction="increasing")
print('From the algorithm aic_elbow_k=',kneed.elbow)    
plt.style.use("seaborn")
plt.plot(range(2, 20), aic)
plt.xticks(range(2, 20))
plt.xlabel("n_components")
plt.ylabel("AIC")
plt.title("AIC vs. k using Gaussian Mixture Models")
plt.show()

aic_elbow_k=kneed.elbow


# In[205]:


# https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
lowest_bic = np.infty
bic = []
n_components_range = range(1, 4)
cv_types = ["spherical", "tied", "diag", "full"]
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type=cv_type
        )
        gmm.fit(x_after_Standardisation)
        bic.append(gmm.bic(x_after_Standardisation))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
clf = best_gmm
bars = []


# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + 0.2 * (i - 2)
    bars.append(
        plt.bar(
            xpos,
            bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
            width=0.2,
            color=color,
        )
    )

plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
plt.title("BIC score per model")
xpos = (
    np.mod(bic.argmin(), len(n_components_range))
    + 0.65
    + 0.2 * np.floor(bic.argmin() / len(n_components_range))
)
plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
spl.set_xlabel("Number of components")
spl.legend([b[0] for b in bars], cv_types)


# In[206]:


# Plot the winner
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(x_after_Standardisation)
for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(x_after_Standardisation[Y_ == i, 0], x_after_Standardisation[Y_ == i, 1], 0.8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)

plt.xticks(())
plt.yticks(())
plt.title(
    f"Selected GMM: {best_gmm.covariance_type} model, "
    f"{best_gmm.n_components} components"
)
plt.subplots_adjust(hspace=0.35, bottom=0.02)
plt.show()


# In[229]:


bic=[]
for k in range(2, 20):
    gm = GaussianMixture(n_components=3, covariance_type='full', init_params='kmeans', random_state=None).fit(x_after_Standardisation)
    gm.bic(x_after_Standardisation) 
    bic.append(gm.bic(x_after_Standardisation))


# In[234]:


kneed2 = KneeLocator(range(2, 20), bic, curve="convex", direction="decreasing")
print('From the algorithm bic_elbow_k=',kneed2.elbow)    
plt.style.use("fivethirtyeight")
plt.plot(range(2, 20), bic)
plt.xticks(range(2, 20))
plt.xlabel("n_components")
plt.ylabel("BIC")
plt.title("BIC vs. k using Gaussian Mixture Models")
plt.show()
bic_elbow_k=kneed2.elbow


# In[210]:


print('\nClassifying the dataset using n_component=aic_elbow_k')
gm = GaussianMixture(n_components=aic_elbow_k,random_state=0,covariance_type='diag').fit(x_after_Standardisation)
y_gm = gm.predict(x_after_Standardisation)
print('Accuracy:Cannot calculate Accuracy Score because the number of classes is not the same as the number of clusters')
conf_mat = confusion_matrix(target, y_gm)
print('Confusion Matrix of AIC')
print(conf_mat)


# In[140]:


print('\nClassifying the dataset using n_component=bic_elbow_k')
gm = GaussianMixture(n_components=aic_elbow_k,random_state=0).fit(x_after_Standardisation)
y_gm = gm.predict(x_after_Standardisation)
print('Accuracy:Cannot calculate Accuracy Score because the number of classes is not the same as the number of clusters')
conf_mat = confusion_matrix(target, y_gm)
print('Confusion Matrix of BIC')
print(conf_mat)


# In[182]:


print('\nClassifying the dataset using n_component=3')
gm = GaussianMixture(n_components=3,random_state=None,covariance_type='diag').fit(x_after_Standardisation)
y_gm = gm.predict(x_after_Standardisation)

print('Accuracy=',accuracy_score(target, y_gm))
conf_mat = confusion_matrix(target, y_gm)
print('Confusion Matrix for n_component 3')
print(conf_mat)

