#!/usr/bin/python
from time import sleep, time

class ProgressMonitor(object):

    def __init__(self, total=100, print_interval=1, msg=None):
        self.tstart = time()
        self.total = total
        self.print_interval = print_interval
        self.progress = 0
        self.pending = 0
        self.msg = msg
        self.print_progress()

    def increment(self, amount=1):
        self.pending += amount
        if ((100*self.pending/self.total)
             >= self.print_interval):
            self.progress += self.pending
            self.pending = 0
            self.print_progress()

    def print_progress(self):
        elapsed = time() - self.tstart
        pc = (100*self.progress/self.total)
        if pc > 0:
            remaining = elapsed*(100. - pc)/pc
            rm_str = format_time(remaining)
        else:
            rm_str = '???'
        if self.msg:
            print '%s: %3d%% (~%s remaining)' % (self.msg, pc, rm_str)
        else:
            print '%3d%% (~%s remaining)' % (pc, rm_str)

def format_time(time):
    if time >= 3600:
        return '%.1f hours' % (time/3600)
    elif time >= 60:
        return '%.1f minutes' % (time/60)
    else:
        return '%.1f seconds' % time

if __name__ == '__main__':
    totaltime = 10.0
    total = 1000
    prog = ProgressMonitor(total, 5, "Testing")
    for i in xrange(total):
        sleep(totaltime/total)
        prog.increment()
