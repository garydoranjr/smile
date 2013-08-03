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
            rank_dict = defaultdict(list)
            for dataset, shuffled in v.items():
                baseline = shuffled.pop(0)
                for s, a in sorted(shuffled.items()):
                    rank_dict[s].append(np.greater(a, baseline).astype(float))

            for s, ranks in sorted(rank_dict.items()):
                key = (t, c, k, i, s, q)
                ranks = np.average(np.vstack(ranks), axis=0)
                line = ','.join(map(str, key) + map(str, ranks.flat))
                print line
                f.write(line + '\n')

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog stats-file winrate-file")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    main(*args, **options)
