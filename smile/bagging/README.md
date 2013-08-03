SMILe: Bagging
==============

This directory contains the code for the bagging experiments of SMILe. These
experiments implement the approach found in:

> Zhou, Zhi-Hua, and Min-Ling Zhang. "Ensembles of multi-instance learners."
> Machine Learning: ECML 2003. Springer Berlin Heidelberg, 2003. 492-502.

Folds and Repetitions
---------------------

Since this experiment generates results that are to be compared to those from
the classification experiment, the same folds should be used. The script
`copy_folds.py` can be used for this purpose. This has already been done for the
included folds, but for new folds this can be done using:

    $ ./src/copy_folds.py folds ../classification/folds/*

Otherwise, the `make_folds.py` script is included to generate new folds for a
dataset.

For this experiment, bootstrap replicates must be generated before the server is
started. This can be accomplished using the `make_reps.py` script as follows:

    $ ./src/make_reps.py -r 100 tiger folds folds

This will make 100 bootstrap replicates of each leave-one-fold-out training set
and put it in the `folds` directory (the redundant final argument is the output
directory). If you later want to increase the number of bootstrap replicates,
you can rerun the script (for example, with `-r 250`), and only the additional
replicates (150 in this example) will be generated.

Configuration
-------------

The configuration file has the following format:

    experiments:
        - classifier: nsk
          dataset: elephant
          reps: 100
          kernel: linear_av
          params:
              C: 1.0

        ...

The "root" of the file contains a list of experiments. Each experiment
specifies:

1. A classifier (`nsk` is for normalized set kernel)
2. A dataset
3. The maximum number of bootstrap replicates to use
4. The kernel used by the base classifier
5. A dictionary of `{ parameter : value }` settings for the base classifier

Statistics
----------

The AUC statistics are stored in a CSV-formatted file with the following
columns:

1. Base Classifier
2. Dataset
3. Kernel
4. Test AUC of classifier trained on bootstrap sample #1
5. Test AUC of ensemble classifier using samples #1-2
5. Test AUC of ensemble classifier using samples #1-3
6. ...

Thus, the final column contains the test AUC of the full ensemble classifier.
