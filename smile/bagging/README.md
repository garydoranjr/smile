SMILe: Bagging
==============

This directory contains the code for the bagging experiments of SMILe. These
experiments implement the approach found in:

> Zhou, Zhi-Hua, and Min-Ling Zhang. "Ensembles of multi-instance learners."
> Machine Learning: ECML 2003. Springer Berlin Heidelberg, 2003. 492-502.

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
3. The maximum number of bootstrap replicates
4. The kernel used by the base classifier
5. A dictionary of `{ parameter : value }` settings for the base classifier

Results
-------

The results are stored in a CSV-formatted file with the following columns:

1. Base Classifier
2. Dataset
3. Kernel
4. Test AUC of classifier trained on bootstrap sample #1
5. Test AUC of ensemble classifier using samples #1-2
5. Test AUC of ensemble classifier using samples #1-3
6. ...

Thus, the final column contains the test AUC of the full ensemble classifier.
