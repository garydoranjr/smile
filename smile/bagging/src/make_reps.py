#!/usr/bin/env python
import os
import numpy as np

import data
from progress import ProgressMonitor

def main(dataset, folddir, outputdir, reps=0):
    data_dict = data.get_dataset(dataset)
    folds = data.get_folds(folddir, dataset)

    all_bag_ids = set(data_dict.keys())

    progress = ProgressMonitor(total=reps*len(folds),
                               msg='Generating Replicates')

    for f in range(len(folds)):
        test = data.get_fold(folddir, dataset, f)
        bag_ids = np.array(list(all_bag_ids - set(test)))
        n = len(bag_ids)

        for r in range(1, reps + 1):
            rep_path = os.path.join(outputdir, '%s_%04d_%06d.rep' % (dataset, f, r))
            if not os.path.exists(rep_path):
                sample = np.random.randint(n, size=n)
                sampled_bags = bag_ids[sample]
                with open(rep_path, 'w+') as ofile:
                    ofile.write('\n'.join([bid for bid in sampled_bags.flat]))
            progress.increment()

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog [options] dataset folddir outputdir")
    parser.add_option('-r', '--reps', dest='reps',
                      type='int', metavar='REPS', default=0)
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 3:
        parser.print_help()
        exit()
    main(*args, **options)

