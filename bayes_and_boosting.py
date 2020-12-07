#!/usr/bin/python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random

irisX = np.loadtxt('irisX.txt', delimiter=",")
irisY = np.loadtxt('irisY.txt', delimiter=",")

# Compute Prior function without Boosting weights
''' 
def computePrior(labels, W=None):
    Npts = labels.shape[0]

    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    # ==========================
    for i,k in enumerate(classes):
        Nk = np.count_nonzero(labels == k)
        prior[i] = Nk / Npts
    # ==========================

    return prior
'''

# Compute Prior with Boosting Weights
def computePrior(labels, W):
    Npts = labels.shape[0]

    if W is None:
        W = np.ones((Npts, 1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses, 1))

    # ==========================
    for i, k in enumerate(classes):
        class_obs = np.where(labels == k)[0]
        prior[i] = np.sum(W[class_obs]) / np.sum(W[:])
    # ==========================

    return prior


prior = computePrior(irisY, W=None)

# mlParams without Boosting Weights
'''
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # ==========================
    for i,k in enumerate(classes):
        #Counts N_samples in every class
        Nk = np.count_nonzero(labels == k)
        #Gives a vector with observations for each class
        class_obs = np.where(labels==k)[0]

        mu[i] = np.sum(X[class_obs,:],axis=0)/Nk

    for i,k in enumerate(classes):
        Nk = np.count_nonzero(labels == k)
        class_obs = np.where(labels==k)[0]

        #Loop over dimensions in sigma
        for j in range(Ndims):
            suma = 0.0
            #Loop over observations in each class 
            for obs in class_obs:
                suma += (X[obs][j] - mu[i][j]) ** 2
            sigma[i][j][j] = suma/Nk
    # ==========================

    return mu,sigma
'''

# mlParams with boosting weights
def mlParams(X, labels, W):
    assert(X.shape[0] == labels.shape[0])
    Npts, Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts, 1))/float(Npts)

    mu = np.zeros((Nclasses, Ndims))
    sigma = np.zeros((Nclasses, Ndims, Ndims))

    # ==========================
    for i, k in enumerate(classes): 
        # Observations in class k array
        class_obs = np.where(labels == k)[0]
        # Calc mu vec, axis=0 gives colsum
        mu[i] = np.sum(W[class_obs, :] * X[class_obs, :],
                       axis=0) / np.sum(W[class_obs, :])

    for i, k in enumerate(classes):  # Compute sigma
        class_obs = np.where(labels == k)[0]

        for j in range(Ndims):
            suma = 0.0
            for obs in class_obs:
                suma += W[obs] * pow(X[obs][j] - mu[i][j], 2)
            sigma[i][j][j] = suma/np.sum(W[class_obs, :])
    # ==========================

    return mu, sigma


mu, sigma = mlParams(irisX, irisY, W=None)


def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses, Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # ==========================
    for i in range(Npts):
        for j in range(Nclasses):
            logProb[j][i] = (-(1/2)*np.log(np.linalg.det(sigma[j])) -
                             (1/2)*np.linalg.multi_dot([(X[i] - mu[j]), np.linalg.inv(sigma[j]), np.transpose((X[i] - mu[j]))]) +
                             np.log(prior[j]))
                             
    # ==========================

    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb, axis=0)
    return h

# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:

# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Test the Maximum Likelihood estimates
#
# Call `genBlobs` and `plotGaussian` to verify your estimates.

'''
X, labels = genBlobs(centers=5)
mu, sigma = mlParams(X, labels, W=None)
plotGaussian(X, labels, mu, sigma)
'''

# Call the `testClassifier` and `plotBoundary` functions for this part.

#testClassifier(BayesClassifier(), dataset='iris', split=0.7)


#testClassifier(BayesClassifier(), dataset='vowel', split=0.7)


#plotBoundary(BayesClassifier(), dataset='iris', split=0.7)
#plotBoundary(BayesClassifier(), dataset='vowel', split=0.7)


# ## Boosting functions to implement
#
# The lab descriptions state what each function should do.


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts, Ndims = np.shape(X)

    classifiers = []  # append new classifiers to this list
    alphas = []  # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts, 1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # ==========================

        # Get all correctly/wrongly classified observations
        classified_obs = np.where(vote == labels)[0]
        missclassified_obs = np.where(vote != labels)[0]
        epsilon = 0.0001

        # Loop over correctly classified obs -> delta(h^t(x_i)) = 0
        for i in missclassified_obs:
            epsilon += wCur[i]

        # Choose alpha
        alpha = (1/2)*(np.log(1 - epsilon) - np.log(epsilon))
        alphas.append(alpha)

        # Update weights if h^t(x_i) = c_i or not
        wOld = wCur
        # h^t(x_i) = c_i
        for i in classified_obs:
            wCur[i] = wOld[i] * np.exp(-alpha)
        #h^t(x_i) != c_i
        for i in missclassified_obs:
            wCur[i] = wOld[i] * np.exp(alpha)

        # Ensure wCur sum to 1
        wCur /= np.sum(wCur)
        # ==========================

    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points


def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts, Nclasses))

        # ==========================
        for i in range(Ncomps):
            aux = classifiers[i].classify(X)
            for j, label in enumerate(aux):
                votes[j][label] += alphas[i]
        # ==========================

                # one way to compute yPred after accumulating the votes
        return np.argmax(votes, axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(
            self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
#
# Call the `testClassifier` and `plotBoundary` functions for this part.


#testClassifier(BoostClassifier(BayesClassifier(), T=10),
               #dataset='iris', split=0.7)


#testClassifier(BoostClassifier(BayesClassifier(), T=10),
               #dataset='vowel', split=0.7)


#plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris', split=0.7)
#plotBoundary(BoostClassifier(BayesClassifier()), dataset='vowel', split=0.7)


# Now repeat the steps with a decision tree classifier.


#testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)


#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


#testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)


#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)


#plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)


#plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# ## Bonus: Visualize faces classified using boosted decision trees
#
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.


#testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)


#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!


# X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
# xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
# pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
# pca.fit(xTr) # use training data to fit the transform
# xTrpca = pca.transform(xTr) # apply on training data
# xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
#classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
#yPr = classifier.classify(xTepca)
# choose a test point to visualize
#testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
#visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])
