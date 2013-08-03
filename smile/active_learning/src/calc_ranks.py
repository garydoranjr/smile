#!/usr/bin/env python
import os
from collections import defaultdict
import numpy as np

def main(stats_file, rank_file):
    stats_dict = defaultdict(lambda: defaultdict(dict))
    with open(stats_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            t, c, d, k, i, s, q = parts[:7]
            stats = np.array(parts[7:], dtype=float)
            stats_dict[t, c, k, int(i), q][d][int(s)] = stats

    with open(rank_file, 'w+') as f:
        for k, v in sorted(stats_dict.items()):
            t, c, k, i, q = k
            shuffs = set()
            stats = np.dstack([np.vstack([a for s, a in sorted(shuffled.items())
                                            if not shuffs.add(s)])
                               for dataset, shuffled in v.items()])
            ranks = (np.argsort(-stats, axis=0) + 1)
            avg_ranks = np.average(ranks, axis=2)
            for s, ranks in zip(sorted(shuffs), avg_ranks):
                key = (t, c, k, i, s, q)
                line = ','.join(map(str, key) + map(str, ranks.flat))
                print line
                f.write(line + '\n')

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog stats-file rank-file")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    main(*args, **options)
