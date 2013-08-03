SMILe: Active Learning
======================

This directory contains the code for the active learning experiments of SMILe.
The instance-labeled SIVAL datasets are from:

> Settles, Burr, Mark Craven, and Soumya Ray. "Multiple-instance active
> learning." Advances in Neural Information Processing Systems. 2007.

Configuration
-------------

The configuration file has the following format:

    experiments:
        - technique: shuffle_both
          noise: 0.100
          classifier: nsk
          dataset: sival_ajaxorange
          reps: 10
          queries: 25
          initial: [5, 10]
          shuffled: [0, 5, 10]
          kernel: linear_av
          params:
              C: 1.0

        ...

The "root" of the file contains a list of experiments. Each experiment
specifies:

1.  A shuffling technique: either `shuffle_both` (shuffles both positive and
    negative bags) or `shuffle_pos` (shuffles positive bags only)
2.  A worst case noise rate (used to determine shuffled bag size)
3.  A classifier (`nsk` is for normalized set kernel)
4.  A dataset
5.  The number of repetitions for each number of shuffled bags and initial bags
6.  The number of active learning queries to perform
7.  A list of the initial bags to use (of each class)
8.  A list of the number of shuffled bags to add to this dataset (of each class)
9.  The kernel used by the base classifier
10. A dictionary of `{ parameter : value }` settings for the base classifier

Statistics
----------

There are several sets of statistics that can be generated from the results:

### AUC
The AUC statistics are computed using `calc_stats.py`, and are stored in a
CSV-formatted file with the following columns:

1.  Shuffling Technique
2.  Base Classifier
3.  Dataset
4.  Kernel
5.  Initial Bags
6.  Shuffled Bags
7.  Active Learning Queries
8.  Test AUC after 0 Queries
9.  Test AUC after 1 Query
10. ...

Thus, the final column contains the test AUC after all active learning queries.

### Improvement Rate
Also called the "win rate", this is the fraction of the time (across datasets)
that using shuffled bags improved performance over not using shuffled bags. This
is computed using the following command:

    $ ./src/calc_winrate.py aucs.csv winrates.csv

Here you can replace `aucs.csv` with the name of the file containing the AUC
statistics, and `winrate.csv` with the desired name of the output file. The
output file will contain the following columns:

1. Shuffling Technique
2. Base Classifier
3. Kernel
4. Initial Bags
5. Shuffled Bags
6. Active Learning Queries
7. Improvement Rate after 0 Queries
8. Improvement Rate after 1 Query
9. ...

### Wilcoxon Signed-Rank Test
There is a script to calculate the Wilcoxon signed-rank test between the paired
AUC values of the base classifier with and without shuffled bags, across
datasets, at each active learning query. The syntax for the script is:

    $ ./src/calc_wilcoxon.py aucs.csv pvalues.csv

As for improvement rate, `aucs.csv` is the file that contains the AUC
statistics, and `pvalues.csv` can be replaced with the desired output file name.
The output file contains the following columns:

1. Shuffling Technique
2. Base Classifier
3. Kernel
4. Initial Bags
5. Shuffled Bags
6. Active Learning Queries
7. p-value after 0 Queries
8. p-value after 1 Queries
9. ...
