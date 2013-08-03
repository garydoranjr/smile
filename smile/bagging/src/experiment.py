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
    (classifier, dataset, kernel, fold, rep) = key

    print 'Starting task %s...' % str(key)
    print 'Parameters: %s' % str(params)

    data_dict = data.get_dataset(dataset)

    fold_ids = set(data.get_fold(FOLDIR, dataset, fold))
    test_ids = fold_ids
    if rep == 0:
        train_ids = set(data_dict.keys()) - test_ids
    else:
        train_ids = set(data.get_rep(FOLDIR, dataset, fold, rep))

    X_train = [data_dict[bid][0] for bid in train_ids]
    y_train = [data_dict[bid][1] for bid in train_ids]

    X_test = [data_dict[bid][0] for bid in test_ids]
    y_test = [data_dict[bid][1] for bid in test_ids]

    results = {}
    results['stats'] = {}
    results['preds'] = {}
    start = time.time()

    if classifier == 'nsk':
        nsk = SetSVM(SVC, kernel, **params)
        nsk.fit(X_train, y_train)
        predictions = nsk.decision_function(X_test)

    else:
        print 'Technique "%s" not supported' % technique
        callback.quit = True
        return

    results['stats']['time'] = time.time() - start
    for i, y in zip(test_ids, predictions):
        results['preds'][i] = float(y)

    if len(y_test) > 1:
        print 'Test AUC Score: %f' % auc_score(y_test, predictions)

    print 'Finished task %s.' % str(key)
    return results
