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
from active_learning import SVMActiveLearner

FOLDIR = 'folds'

def client_target(task, callback):
    key = task['key']
    params = task['params']
    labeled = task['labeled']
    (technique, classifier, dataset, kernel,
     fold, rep, initial, shuffled, queries) = key

    print 'Starting task %s...' % str(key)
    print 'Parameters: %s' % str(params)

    ids, X, y = data.get_dataset(dataset)
    id_index = {}
    for j, i in enumerate(ids):
        id_index[i] = j

    fold = set(data.get_fold(FOLDIR, dataset, fold))
    test_ids = fold
    train_ids = set(ids) - test_ids
    labeled_ids = set((l[1], l[2]) for l in labeled)
    pool_ids = train_ids - labeled_ids

    X_labeled = defaultdict(list)
    y_labeled = defaultdict(bool)
    for bag, bid, iid, yi in labeled:
        X_labeled[bag].append(X[id_index[bid, iid]])
        y_labeled[bag] |= bool(yi)
    bags_labeled = sorted(X_labeled.keys())
    X_labeled = map(np.vstack, [X_labeled[b] for b in bags_labeled])
    y_labeled = [y_labeled[b] for b in bags_labeled]

    X_pool = defaultdict(list)
    y_pool = defaultdict(bool)
    for bid, iid in pool_ids:
        X_pool[bid].append(X[id_index[bid, iid]])
        y_pool[bid] |= bool(y[id_index[bid, iid]])
    bags_pool = sorted(X_pool.keys())
    X_pool = map(np.vstack, [X_pool[b] for b in bags_pool])
    y_pool = [y_pool[b] for b in bags_pool]

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
        cls = SetSVM(SVC, kernel, **params)
        active = SVMActiveLearner(cls, queries)
        predictions = active.learn(X_labeled, y_labeled, X_pool, y_pool, X_test)
    else:
        print 'Technique "%s" not supported' % technique
        callback.quit = True
        return

    results['stats']['time'] = time.time() - start
    for q, preds in enumerate(predictions):
        results['preds'][q] = {}
        for bid, y in zip(bags_test, preds):
            results['preds'][q][bid] = float(y)

    predictions = np.column_stack(predictions).T
    print predictions.shape
    if len(bags_test) > 1:
        print 'Test AUC Scores:'
        for row in predictions:
            print '\t%f' % auc_score(np.array(y_test), row.reshape((-1,)))

    print 'Finished task %s.' % str(key)
    return results
