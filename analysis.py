# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:51:19 2017

@author: emsuter
"""

# Script for building classifiers on Active Site dataset
# Written by Chris Mathy (cjmathy@berkeley.edu), Jordi Silvestre-Ryan
# (jordisr@berkeley.edu), and Emily Suter (emily.suter@berkeley.edu)
# for CS298a, Spring 2017.

import os
import pandas as pd
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


def preprocess(datafile, load=False):

    if load:
        X = np.load(os.getcwd() + '/X.npy')
        y = np.load(os.getcwd() + '/y.npy')

    else:
        # read the data in as a pandas df
        df = pd.read_csv(datafile, header=0, index_col=0).dropna(axis=0)

        features = df.columns.values.tolist()

        for col in ('res_id',
                    'y_label',
                    'pdb',
                    'chain'):
            features.remove(col)
        X = df[features].values
        y = df['y_label'].values

        # center and normalize the data
        # if std = 0 for a feature, don't normalize (set std = 1)
        X = X - np.mean(X, axis=0)
        std = np.std(X, axis=0)
        for i, sigma in enumerate(std):
            if sigma == 0.0:
                std[i] = 1.
        X = np.divide(X, std)

        np.save(os.getcwd() + '/X.npy', X)
        np.save(os.getcwd() + '/y.npy', y)

    pos_i = np.where(y)[0]
    neg_i = np.where(np.ones(y.shape) - y)[0]

    print(np.shape(pos_i), np.shape(neg_i))

    X_pos = X[pos_i]
    X_neg = X[neg_i]
    y_pos = y[pos_i]
    y_neg = y[neg_i]
    
    np.random.shuffle(X_neg)
    np.random.shuffle(y_neg)    
    
    num_pos=len(X_pos)
    
    X_neg_subset=X_neg[:num_pos]
    y_neg_subset=y_neg[:num_pos]

    X_sub=np.concatenate((X_pos,X_neg_subset), axis=0)
    y_sub=np.concatenate((y_pos,y_neg_subset), axis=0)       

    print(np.shape(X_sub), np.shape(y_sub), num_pos)
    return X_sub, y_sub


def logistic_regression(X, y, outdir):

    model = ['logistic']
    penalty = ['l1', 'l2']
    C = [1e0, 1e-1, 1e-2]
    seed = [42]
    grid = itertools.product(model,
                             penalty,
                             C,
                             seed)

    params = {}

    for combination in grid:
        params['model'] = combination[0]
        params['penalty'] = combination[1]
        params['C'] = combination[2]
        params['seed'] = combination[3]

        run_model(X, y, outdir, **params)


def svm(X, y, outdir):

    model = ['svm']
    C = [1e0, 1e-1, 1e-2]
    kernel = ['linear', 'rbf']
    degree = [3, 4, 5]
    seed = [42]
    grid = itertools.product(model,
                             C,
                             kernel,
                             degree,
                             seed)

    params = {}

    for combination in grid:
        params['model'] = combination[0]
        params['C'] = combination[1]
        params['kernel'] = combination[2]
        params['degree'] = combination[3]
        params['seed'] = combination[4]

        run_model(X, y, outdir, **params)


def random_forest(X, y, outdir):

    model = ['rf']
    n_estimators = [10, 20, 50]
    criterion = ['gini', 'entropy']
    max_depth = [None, 3, 5, 10]
    min_imp_split = [1e-5, 1e-6, 1e-7]
    seed = [42]
    grid = itertools.product(model,
                             n_estimators,
                             criterion,
                             max_depth,
                             min_imp_split,
                             seed)

    params = {}

    for combination in grid:
        params['model'] = combination[0]
        params['n_estimators'] = combination[1]
        params['criterion'] = combination[2]
        params['max_depth'] = combination[3]
        params['min_imp_split'] = combination[4]
        params['seed'] = combination[5]

        run_model(X, y, outdir, **params)


def run_model(X, y, outdir, **params):

    import time

    errors = []
    auprcs = []
    aurocs = []
    prc_fig = plt.figure()
    prc_ax = prc_fig.add_subplot(1, 1, 1)
    roc_fig = plt.figure()
    roc_ax = roc_fig.add_subplot(1, 1, 1)

    # prepare validation splits
    n_splits = 5
    test_size = 0.2
    seed = 42
    splitter = ShuffleSplit(n_splits=n_splits,
                            test_size=test_size,
                            random_state=seed)

    name = ''
    for param in params:
        if param is not 'seed':
            if isinstance(params[param], float):
                name = name + '_' + '{:.0g}'.format(params[param])
            else:
                name = name + '_' + str(params[param])
    outfile = outdir + 'output.txt'
    with open(outfile, 'w+') as f:
        f.write('---New Model---\n-Parameters\n')
        for param in params:
            f.write('{}: {}\n'.format(param, params[param]))
        f.write('\n-5 Folds:\n'.format(n_splits))

    fold = 1
    for train, test in splitter.split(X):
        X_trn = X[train]
        y_trn = y[train]
        X_val = X[test]
        y_val = y[test]

        classifier = build_classifier(**params)
        # classifier.fit(X_trn[0:2000,:], y_trn[0:2000])
        classifier.fit(X_trn, y_trn)

        error, auprc, auroc = analyze(classifier,
                                      X_val,
                                      y_val,
                                      prc_ax,
                                      roc_ax,
                                      **params)
        errors.append(error)
        auprcs.append(auprc)
        aurocs.append(auroc)

        with open(outfile, 'a') as f:
            f.write('Fold {} Error: {}\n'.format(fold, error))
            f.write('Fold {} AUPRC: {}\n'.format(fold, auprc))
            f.write('Fold {} AUROC: {}\n'.format(fold, auroc))

        fold += 1

    n_folds = len(errors)
    avg_error = sum(errors)/n_folds
    avg_auprc = sum(auprcs)/n_folds
    avg_auroc = sum(aurocs)/n_folds

    # Write average values to output file
    with open(outfile, 'a') as f:
        f.write("Average Error, {} folds: {}\n".format(n_folds, avg_error))
        f.write("Average AUPRC, {} folds: {}\n".format(n_folds, avg_auprc))
        f.write("Average AUROC, {} folds: {}\n".format(n_folds, avg_auroc))

    # PRC figure
    prc_ax.set_xlabel('Recall')
    prc_ax.set_ylabel('Precision')
    prc_ax.set_xlim([0.0, 1.0])
    prc_ax.set_ylim([0.0, 1.05])
    prc_ax.set_title('PRC{}'.format(name))
    prc_fig.savefig(outdir + 'PRC{}'.format(name)+".png")

    # ROC figure
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_xlim([0.0, 1.0])
    roc_ax.set_ylim([0.0, 1.05])
    roc_ax.set_title('ROC{}'.format(name))
    roc_fig.savefig(outdir + 'ROC{}'.format(name)+".png")

    return


def build_classifier(**params):

    if params['model'] is'logistic':
        print ("logistic")
        return LogisticRegression(penalty=params['penalty'],
                                  C=params['C'],
                                  random_state=params['seed'])

    if params['model'] is 'svm':
        return SVC(C=params['C'],
                   kernel=params['kernel'],
                   degree=params['degree'],
                   random_state=params['seed'])

    if params['model'] is 'rf':
        return RandomForestClassifier(n_estimators=params['n_estimators'],
                                      criterion=params['criterion'],
                                      max_depth=params['max_depth'],
                                      min_imp_split=params['min_imp_split'],
                                      random_state=params['seed'])

    else:
        print("no model specified")
        return


def analyze(classifier, X_val, y_val, prc_ax, roc_ax, **params):

    y_predict = classifier.predict(X_val)

    # Error (1-Accuracy)
    error = classifier.score(X_val, y_val)

    # Precision-Recall
    auprc = prc_score(y_val, y_predict)
    precision, recall, thresholds = prc(y_val, y_predict)
    prc_ax.plot(recall, precision, label='AUC={}'.format(auprc))

    # Receiver Operating Characteristics
    fpr, tpr, thr = roc(y_val, y_predict, pos_label=1)
    auroc = roc_score(fpr, tpr)
    roc_ax.plot(fpr, tpr, label='AUC={}'.format(auroc))

    return error, auprc, auroc


if __name__ == '__main__':
    datafile = os.getcwd() + '/features.csv'
    outdir = os.getcwd() + '/output/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    X, y = preprocess(datafile, load=True)
    # X, y = preprocess(datafile, load=False)
    logistic_regression(X, y, outdir)
    svm(X, y, outdir)
    random_forest(X, y, outdir)
