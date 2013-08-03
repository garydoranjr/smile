#!/usr/bin/env python
import os
import glob
import numpy as np
from collections import defaultdict
from scipy.io import savemat

from progress import ProgressMonitor
from inout import parse_c45

def main(sival_dir, outputfile):
    names_files = glob.glob(os.path.join(sival_dir, '*.names'))
    classes = sorted([os.path.basename(nf[:-6]) for nf in names_files])

    mat = {}
    mat['class_names'] = np.array(classes)

    data = None
    reverse_index = {}
    progress = ProgressMonitor(total=len(classes), msg='Getting class labels')
    for i, clazz in enumerate(classes, 1):
        exset = parse_c45(clazz, sival_dir)
        if data is None:
            data = np.array(exset.to_float())[:, 2:-1]
            inst_classes = np.zeros(len(exset))
            index = [(ex[0], ex[1]) for ex in exset]
            for j, key in enumerate(index):
                reverse_index[key] = j

        for ex in exset:
            inst_classes[reverse_index[(ex[0], ex[1])]] += i*ex[-1]
        progress.increment()

    mat['instance_ids'] = np.array(index)
    mat['X'] = data
    mat['y'] = inst_classes

    savemat(outputfile, mat)

if __name__ == '__main__':
    from sys import argv
    main(*argv[1:])
