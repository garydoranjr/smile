"""
Implements the actual client function to run the experiment
"""
import os
import time
import numpy as np
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.metrics import auc_score

import data
from set_svm import SetSVM

FOLDIR = 'folds'

def client_target(task, callback):
    key = task['key']
    params = task['params']
    shuffled_bags = task['shuffled_bags']
    (technique, classifier, dataset, kernel,
     fold, rep, noise, shuffled) = key

    print 'Starting task %s...' % str(key)
    print 'Parameters: %s' % str(params)

    ids, X, y = data.get_dataset(dataset)
    id_index = {}
    for j, i in enumerate(ids):
        id_index[i] = j

    fold = set(data.get_fold(FOLDIR, dataset, fold))
    test_ids = fold
    train_ids = set(ids) - test_ids

    X_train = defaultdict(list)
    y_train = defaultdict(bool)
    for bid, iid in train_ids:
        X_train[bid].append(X[id_index[bid, iid]])
        y_train[bid] |= bool(y[id_index[bid, iid]])
    for bag, bid, iid, yi in shuffled_bags:
        X_train[bag].append(X[id_index[bid, iid]])
        y_train[bag] |= bool(yi)
    bags_train = sorted(X_train.keys())
    X_train = map(np.vstack, [X_train[b] for b in bags_train])
    y_train = [y_train[b] for b in bags_train]

    X_test = defaultdict(list)
    y_test = defaultdict(bool)
    for bid, iid in test_ids:
        X_test[bid].append(X[id_index[bid, iid]])
        y_test[bid] |= bool(y[id_index[bid, iid]])
    bags_test = sorted(X_test.keys())
    X_test = map(np.vstack, [X_test[b] for b in bags_test])
    y_test = [y_test[b] for b in bags_test]

    results = {}
    results['stats'] = {}
    results['preds'] = {}
    start = time.time()

    if classifier == 'nsk':
        nsk = SetSVM(SVC, kernel, **params)
        nsk.fit(X_train, y_train)
        predictions = nsk.decision_function(X_test)

    else:
        print 'Classifier "%s" not supported' % classifier
        callback.quit = True
        return

    results['stats']['time'] = time.time() - start
    for i, y in zip(bags_test, predictions):
        results['preds'][i] = float(y)

    if len(y_test) > 1:
        print 'Test AUC Score: %f' % auc_score(y_test, predictions)

    print 'Finished task %s.' % str(key)
    return results
