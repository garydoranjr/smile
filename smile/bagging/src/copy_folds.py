#!/usr/bin/env python
import os

def main(outputdir, *smilefolds):
    for foldfile in smilefolds:
        with open(foldfile, 'r') as f:
            bag_ids = set(l.strip().split(',')[0] for l in f)
        fold_path = os.path.join(outputdir, os.path.basename(foldfile))
        with open(fold_path, 'w+') as f:
            f.write('\n'.join([bid for bid in bag_ids]))

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog [options] outputdir smile-fold1 smile-fold2 ...")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) < 2:
        parser.print_help()
        exit()
    main(*args, **options)

