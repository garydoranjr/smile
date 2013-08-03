#!/usr/bin/env python
import os
import yaml
from collections import defaultdict
try:
    from yaml import CLoader as Loader
except ImportError:
    print 'Warning: not using CLoader'
    from yaml import Loader

from progress import ProgressMonitor
from server import Task, setup_rep
import data

def get_positive_shuffled(labeled, initial, shuffled):
    lbound = 2*initial
    ubound = 2*initial + shuffled
    bags = defaultdict(list)
    for i, bid, iid, label in labeled:
        if lbound <= i and i < ubound:
            bags[i].append((bid, iid))
    return bags.values()

def count_actual_positive(bags, y_dict):
    i = 0
    for bag in bags:
        if any(y_dict[bid, iid] for bid, iid in bag):
            i += 1
    return i

def main(configfile, folddir, resultsdir):
    with open(configfile, 'r') as f:
        configuration = yaml.load(f)

    # Generate tasks from experiment list
    total = 0
    actual = 0
    prog = ProgressMonitor(total=len(configuration['experiments']),
                           msg='Computing noise')
    for experiment in configuration['experiments']:
        technique = experiment['technique']
        classifier = experiment['classifier']
        dataset = experiment['dataset']

        ids, _, y = data.get_dataset(dataset)
        y_dict = {}
        for (bid, iid), yi in zip(ids, y):
            y_dict[bid, iid] = yi

        folds = data.get_folds(folddir, dataset)
        for f in range(len(folds)):
            for r in range(experiment['reps']):
                for i in experiment['initial']:
                    for s in experiment['shuffled']:
                        labeled = setup_rep(technique,
                                            experiment['noise'],
                                            dataset, f, r, i, s,
                                            folddir, resultsdir)
                        pos_shuffled = get_positive_shuffled(labeled, i, s)
                        total += len(pos_shuffled)
                        actual += count_actual_positive(pos_shuffled, y_dict)

        prog.increment()
        if total > 0:
            print 1 - (float(actual) / total)

    if total > 0:
        print 1 - (float(actual) / total)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile folddir resultsdir")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 3:
        parser.print_help()
        exit()
    main(*args, **options)
