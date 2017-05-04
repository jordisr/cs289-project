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

global seed
seed = 42

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

    X_pos = X[pos_i]
    X_neg = X[neg_i]
    y_pos = y[pos_i]
    y_neg = y[neg_i]

    num_pos = X_pos.shape[0]
    
    neg_multiplier = 5
    np.random.seed(seed)
    np.random.shuffle(X_neg)
    X_neg_subset = X_neg[:neg_multiplier*num_pos]
    y_neg_subset = y_neg[:neg_multiplier*num_pos]

    X = np.concatenate((X_pos, X_neg_subset), axis=0)
    y = np.concatenate((y_pos, y_neg_subset), axis=0)       

    return X, y


def logistic_regression(X, y, outdir):

    model = ['logistic']
    penalty = ['l1', 'l2']
    C = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    grid = itertools.product(model,
                             penalty,
                             C)
    params_list = ['model',
                   'penalty',
                   'C']
    params = {}
    for combination in grid:
        for i, param in enumerate(params_list):
            params[param] = combination[i]
        run_model(X, y, outdir, **params)


def svm(X, y, outdir):

    model = ['svm']
    C = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    kernel = ['linear', 'rbf']
    grid = itertools.product(model,
                             C,
                             kernel)

    params_list = ['model',
                   'C',
                   'kernel']
    params = {}
    for combination in grid:
        for i, param in enumerate(params_list):
            params[param] = combination[i]
        run_model(X, y, outdir, **params)


def random_forest(X, y, outdir):

    model = ['rf']
    n_estimators = [10, 20, 50]
    criterion = ['gini']
    max_depth = [None, 5, 10, 20]
    min_imp_split = [1e-5, 1e-7, 1e-9]
    grid = itertools.product(model,
                             n_estimators,
                             criterion,
                             max_depth,
                             min_imp_split)
    params_list = ['model',
                   'n_estimators',
                   'criterion',
                   'max_depth',
                   'min_imp_split']
    params = {}
    for combination in grid:
        for i, param in enumerate(params_list):
            params[param] = combination[i]
        run_model(X, y, outdir, **params)


def run_model(X, y, outdir, **params):

    outfile = outdir + 'output_{}.csv'.format(params['model'])
    if not os.path.isfile(outfile):
        with open(outfile, 'w') as f:
            for param in params:
                f.write('{},'.format(param))
            f.write('Error,AUPRC,AUROC,Fold\n')            

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
    splitter = ShuffleSplit(n_splits=n_splits,
                            test_size=test_size,
                            random_state=seed)

    fold = 1
    for train, test in splitter.split(X):
        X_trn = X[train]
        y_trn = y[train]
        X_val = X[test]
        y_val = y[test]

        classifier = build_classifier(**params)
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
            for param in params:
                f.write('{},'.format(params[param]))
            f.write('{},{},{},{}\n'.format(error,
                                           auprc,
                                           auroc,
                                           fold))

        fold += 1

    n_folds = len(errors)
    avg_error = sum(errors)/n_folds
    avg_auprc = sum(auprcs)/n_folds
    avg_auroc = sum(aurocs)/n_folds

    # Write average values to output file
    with open(outfile, 'a') as f:
        for param in params:
            f.write('{},'.format(params[param]))
        f.write('{},{},{},AVG\n'.format(avg_error,
                                        avg_auprc,
                                        avg_auroc,))

    name = ''
    for param in params:
        if isinstance(params[param], float):
            name = name + '_' + '{:.0g}'.format(params[param])
        else:
            name = name + '_' + str(params[param])

    # PRC figure
    prc_ax.set_xlabel('Recall')
    prc_ax.set_ylabel('Precision')
    prc_ax.set_xlim([0.0, 1.0])
    prc_ax.set_ylim([0.0, 1.05])
    prc_ax.set_title('PRC{}'.format(name))
    prc_fig.savefig(outdir + 'PRC{}'.format(name)+".png")
    plt.close(prc_fig)

    # ROC figure
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_xlim([0.0, 1.0])
    roc_ax.set_ylim([0.0, 1.05])
    roc_ax.set_title('ROC{}'.format(name))
    roc_fig.savefig(outdir + 'ROC{}'.format(name)+".png")
    plt.close(roc_fig)

    return


def build_classifier(**params):

    if params['model'] is'logistic':
        print ('logistic')
        return LogisticRegression(penalty=params['penalty'],
                                  C=params['C'],
                                  random_state=seed,
                                  class_weight='balanced')

    if params['model'] is 'svm':
        print ('svm')
        return SVC(C=params['C'],
                   kernel=params['kernel'],
                   random_state=seed,
                   class_weight='balanced')

    if params['model'] is 'rf':
        print ('rf')
        return RandomForestClassifier(n_estimators=params['n_estimators'],
                                      criterion=params['criterion'],
                                      max_depth=params['max_depth'],
                                      min_impurity_split=params['min_imp_split'],
                                      random_state=seed,
                                      class_weight='balanced')

    else:
        print("no model specified")
        return


def analyze(classifier, X_val, y_val, prc_ax, roc_ax, **params):


    #y_predict = classifier.predict(X_val)
    if params['model'] is 'svm' or params['model'] is 'logistic':
        y_predict = classifier.decision_function(X_val)
    else:
        y_predict = classifier.predict_proba(X_val)[:,1]
        print(np.shape(y_predict))

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
    # svm(X, y, outdir)
    random_forest(X, y, outdir)
