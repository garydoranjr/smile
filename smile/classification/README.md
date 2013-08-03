SMILe: Classification
=====================

This directory contains the code for the classification experiments of SMILe.

Configuration
-------------

The configuration file has the following format:

    experiments:
        - technique: shuffle_both
          classifier: nsk
          dataset: tiger
          reps: 10
          shuffled: [0, 25, 50, 75, 100]
          noise: [0.1, 0.3, 0.5]
          kernel: linear_av
          params:
              C: 1.0

        ...

The "root" of the file contains a list of experiments. Each experiment
specifies:

1. A shuffling technique: either `shuffle_both` (shuffles both positive and
   negative bags) or `shuffle_pos` (shuffles positive bags only)
2. A classifier (`nsk` is for normalized set kernel)
3. A dataset
4. The number of repetitions for each number of shuffled bags and noise level
5. A list of the number of shuffled bags to add to this dataset
6. A list of the noise levels to use when choosing shuffled bag size
7. The kernel used by the base classifier
8. A dictionary of `{ parameter : value }` settings for the base classifier

Results
-------

The results are stored in a CSV-formatted file with the following columns:

1. Shuffling Technique
2. Base Classifier
3. Dataset
4. Kernel
5. Worst Case Noise Level
6. Number of Shuffled Bags
7. Average AUC (across repetitions pooled across test folds)
