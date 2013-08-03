#!/usr/bin/env python
import os
import yaml
import time
import random
import cherrypy
from cherrypy import expose, HTTPError
from threading import RLock
from collections import defaultdict
from random import shuffle
import numpy as np

from data import get_folds

PORT = 2118
DEFAULT_TASK_EXPIRE = 120 # Seconds
TEMPLATE = """
<html>
<head>
  <META HTTP-EQUIV="REFRESH" CONTENT="60">
  <title>Experiment Status</title>
  <style type="text/css">
    table.status {
      border-width: 0px;
      border-spacing: 0px;
      border-style: none;
      border-color: black;
      border-collapse: collapse;
      background-color: white;
      margin-left: auto;
      margin-right: auto;
    }
    table.status td {
        border-width: 1px;
        padding: 1px;
        border-style: solid;
        border-color: black;
        text-align: center;
    }
    table.summary {
      border-width: 0px;
      border-spacing: 0px;
      border-style: none;
      border-color: none;
      border-collapse: collapse;
      background-color: white;
      margin-left: auto;
      margin-right: auto;
    }
    table.summary td {
        border-width: 0px;
        padding: 3px;
        border-style: none;
        border-color: black;
        text-align: center;
        width: 50px;
    }
    td.tech { width: 50px; }
    td.done {
      background-color: green;
    }
    td.pending {
      background-color: yellow;
    }
    td.failed {
      background-color: red;
    }
    td.na {
      background-color: gray;
    }
  </style>
</head>
<body>
<h1>Time Remaining: %s</h1>
%s
</body>
</html>
"""

def plaintext(f):
    f._cp_config = {'response.headers.Content-Type': 'text/plain'}
    return f

class ExperimentServer(object):

    def __init__(self, tasks, render, handle,
                 task_expire=DEFAULT_TASK_EXPIRE):
        self.status_lock = RLock()
        self.tasks = tasks
        self.handle = handle
        self.render = render
        self.task_expire = task_expire

        self.unfinished = set(self.tasks.items())

    def clean(self):
        with self.status_lock:
            self.unfinished = filter(lambda x: (not x[1].finished),
                                     self.unfinished)
            for key, task in self.unfinished:
                if (task.in_progress and
                    task.staleness() > self.task_expire):
                    task.quit()

    @expose
    def index(self):
        with self.status_lock:
            self.clean()
            return self.render(self.tasks)

    @plaintext
    @expose
    def request(self):
        with self.status_lock:
            self.clean()
            # Select a job to perform
            unfinished = list(self.unfinished)
            shuffle(unfinished)
            candidates = sorted(unfinished, key=lambda x: x[1].priority())
            if len(candidates) == 0:
                raise HTTPError(404)
            key, task = candidates.pop(0)
            task.ping()
        arguments = {'key': key, 'params': task.params}
        return yaml.dump(arguments)

    @plaintext
    @expose
    def update(self, key_yaml=None):
        try:
            key = yaml.load(key_yaml)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.ping()
            else:
                # Someone else already finished
                raise HTTPError(410)
        return "OK"

    @plaintext
    @expose
    def quit(self, key_yaml=None):
        try:
            key = yaml.load(key_yaml)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.quit()
            else:
                # Someone else already finished
                raise HTTPError(410)
        return "OK"

    @plaintext
    @expose
    def fail(self, key_yaml=None):
        try:
            key = yaml.load(key_yaml)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.fail()
            else:
                # Someone else already finished
                raise HTTPError(410)
        return "OK"

    @plaintext
    @expose
    def submit(self, key_yaml=None, sub_yaml=None):
        try:
            key = yaml.load(key_yaml)
            submission = yaml.load(sub_yaml)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                self.handle(key, task, submission)
                task.finish()
        return "OK"

class Task(object):

    def __init__(self, classifier, dataset, kernel,
                 fold, rep, params=None):
        self.classifier = classifier
        self.dataset = dataset
        self.kernel = kernel
        self.fold = fold
        self.rep = rep
        self.params = params

        self.last_checkin = None
        self.finished = False
        self.failed = False
        self.in_progress = False

        self.finish_time = None

    def filebase(self, ext):
        return ('%s_%s_%s_%d_%d.%s' %
                (self.classifier, self.dataset, self.kernel,
                 self.fold, self.rep, ext))

    def ping(self):
        if not self.finished:
            self.in_progress = True
            self.last_checkin = time.time()

    def quit(self):
        if not self.finished:
            self.in_progress = False
            self.last_checkin = None

    def fail(self):
        if not self.finished:
            self.failed = True
            self.in_progress = False

    def staleness(self):
        return time.time() - self.last_checkin

    def priority(self):
        return 1000*int(self.in_progress) + 100*int(self.failed)

    def finish(self):
        self.finished = True
        self.in_progress = False
        self.failed = False
        self.finish_time = time.time()

def time_remaining_estimate(tasks, alpha=0.1):
    to_go = float(len([task for task in tasks if not task.finished]))
    finish_times = sorted([task.finish_time for task in tasks if task.finished])
    ewma = 0.0
    for interarrival in np.diff(finish_times):
        ewma = alpha*interarrival + (1.0 - alpha)*ewma

    if ewma == 0:
        return '???'

    remaining = to_go * ewma
    if remaining >= 604800:
        return '%.1f weeks' % (remaining/604800)
    elif remaining >= 86400:
        return '%.1f days' % (remaining/86400)
    elif remaining >= 3600:
        return '%.1f hours' % (remaining/3600)
    elif remaining >= 60:
        return '%.1f minutes' % (remaining/60)
    else:
        return '%.1f seconds' % remaining

def render(tasks):
    # Get dimensions
    dims = [set() for i in range(3)]
    for key in tasks.keys():
        dims[0].add(key[0])
        dims[1].add(key[1])
        dims[2].add(key[2])
    techniques, datasets, kernels = map(sorted, dims)

    time_est = time_remaining_estimate(tasks.values())

    reindexed = defaultdict(list)
    for k, v in tasks.items():
        key = (k[0], k[1], k[2])
        reindexed[key].append(v)

    tasks = reindexed

    table = '<table class="status">'
    # Technique header row
    table += '<tr><td style="border:0" rowspan="1" colspan="2"></td>'
    for technique in techniques:
        table += ('<td class="tech">%s</td>' % technique)
    table += '</tr>\n'

    # Data rows
    for dataset in datasets:
        table += ('<tr><td rowspan="%d" class="data">%s</td>' % (len(kernels), dataset))
        first_kernel = True
        for kernel in kernels:
            if first_kernel:
                first_kernel = False
            else:
                table += '<tr>'
            table += ('<td class="kernel">%s</td>' % kernel)
            for technique in techniques:
                key = (technique, dataset, kernel)
                title = ('%s, %s, %s' % key)
                if key in tasks:
                    table += ('<td style="padding: 0px;">%s</td>' % render_task_summary(tasks[key]))
                else:
                    table += ('<td class="na" title="%s"></td>' % title)
            table += '</tr>\n'

    table += '</table>'
    return (TEMPLATE % (time_est, table))

def render_task_summary(tasks):
    n = float(len(tasks))
    failed = 0
    finished = 0
    in_progress = 0
    waiting = 0
    for task in tasks:
        if task.finished:
            finished += 1
        elif task.failed:
            failed += 1
        elif task.in_progress:
            in_progress += 1
        else:
            waiting += 1

    if n == finished:
        table = '<table class="summary"><tr>'
        table += ('<td class="done" title="Finished">D</td>')
        table += ('<td class="done" title="Finished">O</td>')
        table += ('<td class="done" title="Finished">N</td>')
        table += ('<td class="done" title="Finished">E</td>')
        table += '</tr></table>'
    else:
        table = '<table class="summary"><tr>'
        table += ('<td title="Waiting">%.2f%%</td>' % (100*waiting/n))
        table += ('<td class="failed" title="Failed">%.2f%%</td>' % (100*failed/n))
        table += ('<td class="pending" title="In Progress">%.2f%%</td>' % (100*in_progress/n))
        table += ('<td class="done" title="Finished">%.2f%%</td>' % (100*finished/n))
        table += '</tr></table>'
    return table

def main(configfile, folddir, resultsdir):
    with open(configfile, 'r') as f:
        configuration = yaml.load(f)

    # Generate tasks from experiment list
    tasks = {}
    for experiment in configuration['experiments']:
        classifier = experiment['classifier']
        dataset = experiment['dataset']
        folds = get_folds(folddir, dataset)
        for f in range(len(folds)):
            for r in range(experiment['reps'] + 1):
                        key = (classifier, dataset,
                               experiment['kernel'], f, r)
                        kwargs = {}
                        kwargs['params'] = experiment['params']
                        task = Task(*key, **kwargs)
                        tasks[key] = task

    # Mark finished tasks
    for task in tasks.values():
        predfile = os.path.join(resultsdir, task.filebase('preds'))
        if os.path.exists(predfile):
            task.finish()

    def handle(key, task, submission):
        if 'stats' in submission:
            sfile = os.path.join(resultsdir, task.filebase('stats'))
            with open(sfile, 'w+') as f:
                f.write(yaml.dump(submission['stats'], default_flow_style=False))

        pfile = os.path.join(resultsdir, task.filebase('preds'))
        with open(pfile, 'w+') as f:
            f.write(yaml.dump(submission['preds'], default_flow_style=False))

    server = ExperimentServer(tasks, render, handle)
    cherrypy.config.update({'server.socket_port': PORT,
                            'server.socket_host': '0.0.0.0'})
    cherrypy.quickstart(server)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile folddir resultsdir")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 3:
        parser.print_help()
        exit()
    main(*args, **options)
