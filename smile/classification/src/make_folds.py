#!/usr/bin/env python
import os
import numpy as np
from collections import defaultdict
from sklearn.cross_validation import StratifiedKFold

from data import get_dataset

def main(dataset, outputdir, outerfolds=0):
    ids, X, y = get_dataset(dataset)

    bag_index = defaultdict(list)
    classes = defaultdict(list)
    for (bid, iid), yi in zip(ids, y.flat):
        bag_index[bid].append(iid)
        classes[bid].append(yi)
    bag_ids = np.unique(np.array(ids)[:, 0])
    Y = [any(classes[bid]) for bid in bag_ids.flat]

    n = len(bag_ids)
    if outerfolds <= 0:
        outerfolds = n

    def fold_path(o):
        return os.path.join(outputdir, '%s_%04d.fold' % (dataset, o))

    for o, (trn_fold, fold) in enumerate(StratifiedKFold(Y, outerfolds)):
        with open(fold_path(o), 'w+') as f:
            for bid in bag_ids[fold].flat:
                for iid in bag_index[bid]:
                    f.write('%s,%s\n' % (bid, iid))

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog [options] dataset outputdir")
    parser.add_option('-o', '--outer-folds', dest='outerfolds',
                      type='int', metavar='FOLDS', default=0)
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    main(*args, **options)

