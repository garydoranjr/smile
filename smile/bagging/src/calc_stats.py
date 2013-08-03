#!/usr/bin/env python
import os
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    print 'Warning: not using CLoader'
    from yaml import Loader
import numpy as np
from sklearn.metrics import auc_score
from collections import defaultdict

from server import Task
import data

def true_and_pred(y_dict, preds):
    y_true = []
    y_pred = []
    for bid in preds.keys():
        y_true.append(y_dict[bid])
        y_pred.append(preds[bid])
    return map(np.array, (y_true, y_pred))

def get_preds(key, task_list, bag_ids):
    predictions = dict()
    for task in task_list:
        with open(task.predfile, 'r') as f:
            predictions.update(yaml.load(f, Loader=Loader))

    return np.array([predictions[bid] for bid in bag_ids])

def main(configfile, folddir, resultsdir, outputfile):
    with open(configfile, 'r') as f:
        configuration = yaml.load(f)

    # Generate tasks from experiment list
    tasks = {}
    for experiment in configuration['experiments']:
        classifier = experiment['classifier']
        dataset = experiment['dataset']
        folds = data.get_folds(folddir, dataset)
        for f in range(len(folds)):
            for r in range(experiment['reps']):
                key = (classifier, dataset,
                       experiment['kernel'], f, r)
                task = Task(*key)
                tasks[key] = task

    # Mark finished tasks
    for task in tasks.values():
        predfile = os.path.join(resultsdir, task.filebase('preds'))
        task.predfile = predfile
        if os.path.exists(predfile):
            task.finish()

    reindexed = defaultdict(lambda: defaultdict(list))
    for (c, d, k, f, r), task in tasks.items():
        reindexed[(c, d, k)][r].append(task)

    existing_keys = set()
    if os.path.exists(outputfile):
        with open(outputfile, 'r') as f:
            for line in f:
                c, d, k = line.strip().split(',')[:3]
                existing_keys.add((c, d, k))

    with open(outputfile, 'a+') as f:
        rep_aucs = defaultdict(list)
        for key, reps in sorted(reindexed.items()):
            if key in existing_keys:
                print 'Skipping %s (already finished)...' % str(key)
                continue
            data_dict = data.get_dataset(key[1])
            bag_ids = sorted(data_dict.keys())
            y_true = [data_dict[bid][1] for bid in bag_ids]

            predictions = []
            for rep, task_list in sorted(reps.items()):
                if all(task.finished for task in task_list):
                    predictions.append(get_preds(key, task_list, bag_ids))
                else:
                    break
            if len(predictions) != len(reps):
                print 'Skipping %s (incomplete)...' % str(key)
                continue
            predictions = np.vstack(predictions)
            # We want a cumulative average, but doesn't matter for AUC
            cumpreds = np.cumsum(predictions, axis=0)
            aucs = [auc_score(y_true, cp) for cp in cumpreds]
            line = ','.join(map(str, key) + map(str, aucs))
            print line
            f.write(line + '\n')

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile folddir resultsdir outputfile")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 4:
        parser.print_help()
        exit()
    main(*args, **options)
