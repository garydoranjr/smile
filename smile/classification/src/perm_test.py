#!/usr/bin/env python
import os
import sys
import yaml
from itertools import combinations
from collections import defaultdict
try:
    from yaml import CLoader as Loader
except ImportError:
    print >> sys.stderr, 'Warning: not using CLoader'
    from yaml import Loader
from yard.data import BinaryClassifierData
from yard.significance import PairedPermutationTest

import data

def main(*pred_files):
    pred_dict = {}
    for pred_file in pred_files:
        with open(pred_file, 'r') as f:
            pred_dict.update(yaml.load(f))

    for key1, key2 in combinations(sorted(pred_dict.keys()), 2):
        if key1[2] != key2[2]: continue
        if key1[-2] != key2[-2]: continue
        if not ((key1[-1] == 0) or (key2[-1] == 0)): continue
        if not ((key1[-2] in (0.1, 0.3, 0.5)) or
                (key2[-2] in (0.1, 0.3, 0.5))): continue

        ids, _, y = data.get_dataset(key1[2])
        y_dict = defaultdict(bool)
        for (bid, iid), yi in zip(ids, y):
            y_dict[bid] |= bool(yi)

        preds1 = pred_dict[key1]
        preds2 = pred_dict[key2]
        bag_ids = preds1.keys()
        y_true = [y_dict[b] for b in bag_ids]
        preds1 = [preds1[b] for b in bag_ids]
        preds2 = [preds2[b] for b in bag_ids]
        data1 = BinaryClassifierData(zip(preds1, y_true))
        data2 = BinaryClassifierData(zip(preds2, y_true))
        ppt = PairedPermutationTest(num_repetitions=10000)
        diff, pval = ppt.test(data1, data2)
        print '%s vs %s:' % (str(key1), str(key2))
        print 'diff: %f    p: %f' % (diff, pval)
        print

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog pred_files ...")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    main(*args, **options)
