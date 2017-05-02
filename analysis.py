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
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import average_precision_score as prc_score
from sklearn.metrics import roc_curve as roc
from sklearn.metrics import auc as roc_score


def preprocess(datafile):

    # read in dataset
    with open(datafile, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        dataset = list(reader)

    # convert dataset to numpy array. Note that this includes labels!!
    # remove pdb id and name so they don't appear in the array
    n = len(dataset)
    d = len(dataset[0])
    X = np.array(dataset).reshape((n, d))

    # separate labels from sample points
    y = X[:, -1]

    # remove bookkeeping features (chain, pdb, res_id)
    X = np.concatenate((X[:, 0], X[:, 2:-3]), axis=1)

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
    splitter = ShuffleSplit(n_splits=n_splits,
                            test_size=test_size,
                            random_state=seed)

    return X, y, splitter


def logistic_regression(X, y, splitter):

    model = 'logistic'
    penalty = ['l1', 'l2']
    C = [1.0, 0.1, 0.01]
    seed = 42
    grid = model, penalty, C, seed

    params = {}

    for combination in itertools.product(grid):
        params['model'] = combination[0]
        params['penalty'] = combination[1]
        params['C'] = combination[2]
        params['seed'] = combination[3]

        run_model(X, y, splitter, **params)


def svm(X, y, splitter):

    model = 'svm'
    C = [1.0, 0.1, 0.01]
    kernel = ['linear', 'rbf']
    degree = [3, 4, 5]
    seed = 42
    grid = model, C, kernel, degree, seed

    params = {}

    for combination in itertools.product(grid):
        params['model'] = combination[0]
        params['C'] = combination[1]
        params['kernel'] = combination[2]
        params['degree'] = combination[3]
        params['seed'] = combination[4]

        run_model(X, y, splitter, **params)


def random_forest(X, y, splitter):

    model = 'rf'
    n_estimators = [10, 20, 50]
    criterion = ['gini', 'entropy']
    max_depth = [None, 3, 5, 10]
    min_imp_split = [1e-5, 1e-6, 1e-7]
    seed = 42
    grid = model, n_estimators, criterion, max_depth, min_imp_split, seed

    params = {}

    for combination in itertools.product(grid):

        params['model'] = combination[0]
        params['n_estimators'] = combination[1]
        params['criterion'] = combination[2]
        params['max_depth'] = combination[3]
        params['min_imp_split'] = combination[4]
        params['seed'] = combination[5]

        run_model(X, y, splitter, **params)


def run_model(X, y, splitter, **params):

    errors = []
    auprcs = []
    aurocs = []

    for train, test in splitter.split(X):

        X_trn = X[train]
        y_trn = y[train]
        X_val = X[test]
        y_val = y[test]

        classifier = build_classifier(**params)

        classifier.fit(X_trn, y_trn)  # train

        error, auprc, auroc = analyze(classifier, X_val, y_val, **params)
        errors.append(error)
        auprcs.append(auprc)
        aurocs.append(auroc)

    # fix: figure out how to report
    avg_error = errors/len(errors)
    avg_auprc = auprcs/len(auprcs)
    avg_auroc = aurocs/len(aurocs)

    return


def build_classifier(**params):

    if params['model'] == 'logistic':
        return LogisticRegression(penalty=params['penalty'],
                                  C=params['C'],
                                  random_state=params['seed'])

    if params['model'] == 'svm':
        return SVC(C=params['C'],
                   kernel=params['kernel'],
                   degree=params['degree'],
                   random_state=params['seed'])

    if params['model'] == 'rf':
        return RandomForestClassifier(n_estimators=params['n_estimators'],
                                      criterion=params['criterion'],
                                      max_depth=params['max_depth'],
                                      min_imp_split=params['min_imp_split'],
                                      random_state=params['seed'])

    else:
        print("no model specificed")  # fix: add actual error checking?
        return


# fix
def analyze(clfr, model, X_val, y_val, **params):

    y_predict = clfr.predict(X_val)

    # precision-recall curve
    precision, recall, thresholds = prc(y_val, y_predict)
    plot_prc(precision, recall, thresholds, **params)  # FIX?

    # ROC curve
    fpr, tpr, thresholds = roc(y_val, y_predict)

    # compute error, auprc, and auroc
    error = clfr.score(X_val, y_val)
    auprc = prc_score(y_val, predictions)
    auroc = roc_score(fpr, tpr)
    return


# # fix
# def plot_prc(precision, recall, thresholds, **params):

#     # Plot Precision-Recall curve
#     plt.clf()
#     plt.plot(recall[0], precision[0], lw=lw, color='navy',
#              label='Precision-Recall curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0.0, 1.0])
#     plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
#     plt.legend(loc="lower left")
#     plt.show()

#     return


# # fix
# def plot_roc(tpr, fpr, thr, **params):

    return


if __name__ == '__main__':
    datafile = os.getcwd() + '/big.csv'
    X, y, splitter = preprocess(datafile)
    logistic_regression(X, y, splitter)
    svm(X, y, splitter)
    random_forest(X, y, splitter)
