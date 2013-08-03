#!/usr/bin/env python
import os
import numpy as np
from collections import defaultdict
from sklearn.cross_validation import StratifiedKFold

from data import get_dataset

def main(dataset, outputdir, folds=0):
    data_dict = get_dataset(dataset)
    bag_ids = np.array(data_dict.keys())
    Y = [data_dict[bid][1] for bid in bag_ids.flat]

    n = len(bag_ids)
    if folds <= 0:
        folds = n

    for f, (trn_fold, fold) in enumerate(StratifiedKFold(Y, folds)):
        fold_path = os.path.join(outputdir, '%s_%04d.fold' % (dataset, f))
        with open(fold_path, 'w+') as f:
            f.write('\n'.join([bid for bid in bag_ids[fold].flat]))

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog [options] dataset outputdir")
    parser.add_option('-f', '--folds', dest='folds',
                      type='int', metavar='FOLDS', default=0)
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    main(*args, **options)

