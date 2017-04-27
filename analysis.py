# Script for building classifiers on Active Site dataset
# Written by Chris Mathy (cjmathy@berkeley.edu), Jordi Silvestre-Ryan
# (jordisr@berkeley.edu), and Emily Suter (emily.suter@berkeley.edu)
# for CS298a, Spring 2017.


# TO-DO: (27 Apr)
# (1)
# aggregate prc and score values for each split into avg across splits.
# plot avg prc and table with avg score values for each hyperparameter
# combo. do this for each of the three machine learning methods
# (2)
# make sure data import is correct once featurize.py output is finalized
# (3)
# finalize hyperparameter choices for each method
# (4)
# abstract code with a couple of functions: one for preprocessing, and one
# for each ML method, most likely
# (5)
# Comment code

import os
import csv
import numpy as np
import itertools

from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics.precision_recall_curve as prc


def run():

    cwd = os.getcwd()
    datafile = cwd + '/../dataset.csv'

    # read in dataset
    with open(datafile, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader)
        dataset = list(reader)

    # convert dataset to numpy array. Note that this includes labels!!
    n = len(dataset)
    d = len(dataset[0])
    X = np.array(dataset).reshape((n, d))

    # separate labels from sample points
    y = X[:, -1]
    X = X[:, :-1]
    d = d-1

    # center the data
    X = X - np.mean(X, axis=0)

    # normalize the training and test data
    # if std is 0 for a feature, set std = 1. We do this because this feature
    # is the same for all training points, so we don't need to normalize it.
    std = np.std(X, axis=0)
    for i, sigma in enumerate(std):
        if sigma == 0.0:
            std[i] = 1.
    X = np.divide(X, std)

    # prepare validation splits
    n_splits = 5
    test_size = 0.2
    seed = 42
    rs = ShuffleSplit(n_splits=n_splits,
                      test_size=test_size,
                      random_state=seed)

    # ---Logistic Regression---

    # initialize parameter grid
    penalty = ['l1', 'l2']
    C = [1.0, 0.1, 0.01]
    grid = penalty, C

    for parameters in itertools.product(grid):
        p, c = parameters
        clfr_log = LogisticRegression(penalty=p,
                                      C=c,
                                      random_state=seed)

        for train, test in rs.split(X):
            X_trn = X[train]
            y_trn = y[train]
            X_val = X[test]
            y_val = y[test]

            clfr_log.fit(X_trn, y_trn)
            score = clfr_log.score(X_val, y_val)
            z_val = clfr_log.predict(X_val)
            precision, recall, thresholds = prc(y_val, z_val)

    # ---Support Vector Machine---

    # initialize parameter grid
    C = [1.0, 0.1, 0.01]
    kernel = ['linear', 'rbf']
    degree = [3, 4, 5]
    grid = C, kernel, degree

    for parameters in itertools.product(grid):
        c, k, d = parameters
        clfr_svm = SVC(C=c,
                       kernel=k,
                       degree=d,
                       random_state=seed)

        for train, test in rs.split(X):
            X_trn = X[train]
            y_trn = y[train]
            X_val = X[test]
            y_val = y[test]

            clfr_svm.fit(X_trn, y_trn)
            score = clfr_svm.score(X_val, y_val)
            z_val = clfr_svm.predict(X_val)
            precision, recall, thresholds = prc(y_val, z_val)

    # ---Random Forest---

    # initialize parameter grid
    n_estimators = [10, 20, 50]
    criterion = ['gini', 'entropy']
    max_depth = [None, 3, 5, 10]
    min_impurity_split = [1e-5, 1e-6, 1e-7]
    grid = n_estimators, criterion, max_depth, min_impurity_split

    for parameters in itertools.product(grid):
        ne, c, md, mis = parameters
        clfr_rf = RandomForestClassifier(n_estimators=ne,
                                         criterion=c,
                                         max_depth=md,
                                         min_impurity_split=mis,
                                         random_state=seed)
        clfr_rf.fit(X_trn, y_trn)
        score = clfr_rf.score(X_val, y_val)
        z_val = clfr_rf.predict(X_val)
        precision, recall, thresholds = prc(y_val, z_val)


if __name__ == '__main__':

    run()
