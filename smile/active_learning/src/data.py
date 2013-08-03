"""Utility for loading datasets and folds"""
import os
import glob
import numpy as np
from scipy.io import loadmat

from inout import parse_c45

DATA_DIR = 'data'
SIVAL_DATA = 'sival.mat'

CACHE = {}

def get_dataset(dataset_name):
    if not dataset_name in CACHE:
        if dataset_name.startswith('sival'):
            ids, X, y = _get_sival_dataset(dataset_name[6:])
        else:
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

        CACHE[dataset_name] = (ids, X, y)
    return CACHE[dataset_name]

def _get_sival_dataset(dataset_name):
    mat = loadmat(os.path.join(DATA_DIR, SIVAL_DATA))
    class_id = None
    for i, name in enumerate(mat['class_names'], 1):
        if name.strip() == dataset_name:
            class_id = i
            break

    if class_id is None:
        raise Exception('Unknown SIVAL dataset: %s' % dataset_name)

    ids = [(str(i[0].strip()), str(i[1].strip()))
           for i in mat['instance_ids']]
    X = mat['X']
    y = (mat['y'] == class_id)
    return ids, X, y

def get_folds(folddir, dataset):
    regex = os.path.join(folddir, '%s*.fold' % dataset)
    return glob.glob(regex)

def get_fold(folddir, dataset, fold):
    path = os.path.join(folddir, '%s_%04d.fold' % (dataset, fold))
    with open(path, 'r') as f:
        return map(lambda s: tuple(s.strip().split(',')), f)
