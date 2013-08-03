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

def calc_auc_score(key, task_list, y_dict):
    predictions = defaultdict(dict)
    for task in task_list:
        with open(task.predfile, 'r') as f:
            preds = yaml.load(f, Loader=Loader)
        for q, p in preds.items():
            predictions[q].update(p)

    aucs = []
    for q, p in sorted(predictions.items()):
        aucs.append(auc_score(*true_and_pred(y_dict, p)))

    return np.array(aucs)

def main(configfile, folddir, resultsdir, outputfile):
    with open(configfile, 'r') as f:
        configuration = yaml.load(f)

    # Generate tasks from experiment list
    tasks = {}
    for experiment in configuration['experiments']:
        technique = experiment['technique']
        classifier = experiment['classifier']
        dataset = experiment['dataset']
        folds = data.get_folds(folddir, dataset)
        for f in range(len(folds)):
            for r in range(experiment['reps']):
                for i in experiment['initial']:
                    for s in experiment['shuffled']:
                        key = (technique, classifier,
                               dataset,
                               experiment['kernel'],
                               f, r, i, s,
                               experiment['queries'])
                        task = Task(*key)
                        tasks[key] = task

    # Mark finished tasks
    for task in tasks.values():
        predfile = os.path.join(resultsdir, task.filebase('preds'))
        task.predfile = predfile
        if os.path.exists(predfile):
            task.finish()

    reindexed = defaultdict(lambda: defaultdict(list))
    for (t, c, d, k, f, r, i, s, q), task in tasks.items():
        reindexed[(t, c, d, k, i, s, q)][r].append(task)

    existing_keys = set()
    if os.path.exists(outputfile):
        with open(outputfile, 'r') as f:
            for line in f:
                t, c, d, k, i, s, q = line.strip().split(',')[:7]
                existing_keys.add((t, c, d, k, int(i), int(s), int(q)))

    with open(outputfile, 'a+') as f:
        rep_aucs = defaultdict(list)
        for key, reps in sorted(reindexed.items()):
            if key in existing_keys:
                print 'Skipping %s (already finished)...' % str(key)
                continue
            ids, _, y = data.get_dataset(key[2])
            y_dict = defaultdict(bool)
            for (bid, iid), yi in zip(ids, y):
                y_dict[bid] |= bool(yi)

            aucs = []
            for rep, task_list in sorted(reps.items()):
                if all(task.finished for task in task_list):
                    aucs.append(calc_auc_score(key, task_list, y_dict))
                else:
                    break
            if len(aucs) != len(reps):
                print 'Skipping %s (incomplete)...' % str(key)
                continue
            aucs = np.vstack(aucs)
            avg_aucs = np.average(aucs, axis=0)
            line = ','.join(map(str, key) + map(str, avg_aucs.flat))
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
