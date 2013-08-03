"""Utility for loading datasets and folds"""
import os
import glob
import numpy as np

from inout import parse_c45

DATA_DIR = 'data'

CACHE = {}

def get_dataset(dataset_name):
    if not dataset_name in CACHE:
        exset = parse_c45(dataset_name, DATA_DIR)
        raw_data = np.array(exset.to_float())
        X = raw_data[:, 2:-1]
        y = (raw_data[:, -1] == 1).reshape((-1,))
        ids = [(ex[0], ex[1]) for ex in exset]

        # Normalize
        mean = np.average(X, axis=0)
        std = np.std(X, axis=0)
        std[np.nonzero(std == 0.0)] = 1.0
        X = ((X - mean) / std)

        bag_ids = sorted(set(i[0] for i in ids))
        data_dict = {}
        for bid in bag_ids:
            instances = [i for i, (b, iid) in enumerate(ids) if b == bid]
            bag = X[instances, :]
            Y = any(y[instances].flat)
            data_dict[bid] = (bag, Y)

        CACHE[dataset_name] = data_dict
    return CACHE[dataset_name]

def get_folds(folddir, dataset):
    regex = os.path.join(folddir, '%s*.fold' % dataset)
    return glob.glob(regex)

def get_fold(folddir, dataset, fold):
    path = os.path.join(folddir, '%s_%04d.fold' % (dataset, fold))
    with open(path, 'r') as f:
        return map(lambda s: s.strip(), f)

def get_rep(folddir, dataset, fold, rep):
    path = os.path.join(folddir, '%s_%04d_%06d.rep' % (dataset, fold, rep))
    with open(path, 'r') as f:
        return map(lambda s: s.strip(), f)
