SMILe: Shuffled Multiple-Instance Learning
==========================================

Gary Doran (<gary.doran@case.edu>) and Soumya Ray (<sray@case.edu>)

Overview
--------
This package contains the code, datasets, and folds for the experiments found in
the AAAI 2013 paper _SMILe: Shuffled Mutiple-Instance Learning_. There are three
main experiments, each in its own directory:

1. Classification
2. Active Learning
3. Bagging (baseline for comparison)

Each experimental framework follows the same general outline, described below.
More specific information about each experiment can be found in the
corresponding directory.

All experiments are written in Python (version 2.6 or 2.7), and require the
following libraries (versions are included for reference, but the code should
work as long as the versions you use are recent enough):

- numpy (1.7.0)
- scipy (0.11.0)
- PyYAML (3.10)
- CherryPy (3.2.2)
- scikit-learn (0.13.1)

This code has only been tested on UNIX-like systems.

Instructions
------------

Each experiment has the same general directory structure:

- `src` the experiment code
- `config` the experimental configuration
- `folds` the folds/repetitions for each dataset
- `data` the actual datasets

There are several steps necessary to generate results each with a corresponding
command:

1. Generate folds (`make_folds.py`)
2. Start server (`server.py`)
3. Start clients (`client.py`)
4. Calculate statistics (`calc_stats.py`)

Each command must be executed from the experiment directory (not the `src`
directory), since the relative paths to the `folds` and `data` directories are
hard-coded. Each step is described in more detail below.

### Generate Folds

For the included datasets, the folds have already been generated. If you use a
new dataset, C4.5 format is supported. Otherwise, you will need to modify
`data.py` to load your custom format. When you have your dataset, you can
generate folds with a command similar to the following:

    $ ./src/make_folds.py -o 10 tiger folds

This will create 10 "outer" folds for the tiger dataset, and place them in the
`folds` directory. If the `-o` flag is not specified, leave-one-out folds are
generated.

### Start Server

To distribute experiments across machines, a central server is used to
coordinate clients. This server must started and running before clients can run.
The server takes as input the path to a configuration file, which is a
YAML-formatted file holding experiment parameters. The server then generates the
necessary experiments for each combination of parameter settings, datasets, and
folds. For SMILe, this includes generating the shuffled bags (the code to
perform the shuffling is in `shuffling.py`).

Generating the experiments can take several minutes and can use up a bit of
memory if there are many experiments. Therefore, I suggest splitting experiments
into smaller "sub-experiments." The full configuration file for the paper's
experiments are included, along with a "small" version of the configuration file
for testing purposes.

Finally, you will need to make a directory for storing the results of the
experiments (e.g. `results`). The server can then be started using:

    $ ./src/server.py config/small_config.yaml folds results

After the server starts, you can view the progress using a web browser by
visiting `http://localhost:PORT/` where PORT is the port number specified near
the top of `server.py`.

### Start Client

After the server is started, the client can be started to perform experiments.
This can be done via:

    $ ./src/client.py localhost

If the client is started on a remote computer, simply supply the domain name of
the server machine (of course, the server must not be behind a firewall or local
network, etc.):

    $ ./src/client.py my.server.com

The client will request an experiment, perform it, submit the result, and repeat
until all experiments are completed. The actual experiment code is located in
`experiment.py`; this is the code you should modify to add additional
classifiers or to handle parameter settings differently.

After all experiments are finished, the client and server programs can be
terminated.

### Calculate Statistics

As the experiments are completed, results are stored in the specified directory
(called `results` in this example). Results are saved between executions of the
server, so if a server fails, it reload existing results and start from where it
left off. After all results have been generated, the statistics (area under ROC
curve) can be computed. The syntax is similar to the server code:

    $ ./src/calc_stats.py config/small_config.yaml folds results stats.csv

Here, an extra parameter specifies the output file to which results are to be
written. The format of the results in `stats.csv` vary between experiments, and
contain results that are included in the paper.
