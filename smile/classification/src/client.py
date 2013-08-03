#!/usr/bin/env python
import os
import re
import yaml
import time
import traceback
from threading import RLock, Thread
from httplib import HTTPConnection
from urllib import urlencode
from itertools import count

from server import PORT

UPDATE_INTERVAL = 30

class StatusCallback(object):

    def __init__(self, task_string, verbose=True):
        self.verbose = verbose
        self.task_string = task_string
        self.lock = RLock()
        self.retval = None
        self.quit = False
        self.fail = False

def do_request(host, port):
    try:
        connection = HTTPConnection(host, port)
        req = connection.request('GET', '/request')
        resp = connection.getresponse()
        if resp.status == 200:
            body = resp.read()
            task = yaml.load(body)
            return task
        else:
            raise Exception('Got code: %d' % resp.status)
    except Exception as e:
        print 'Could not get task: %s' % e
    finally:
        connection.close()

def do_update(host, port, task, update='update'):
    try:
        params = urlencode({'key_yaml': yaml.dump(task)})
        headers = {"Content-type": "application/x-www-form-urlencoded",
                   "Accept": "text/plain"}
        connection = HTTPConnection(host, port)
        req = connection.request('POST', '/%s' % update, params, headers)
        resp = connection.getresponse()
        if resp.status == 200:
            body = resp.read()
        else:
            raise Exception('Got code: %d' % resp.status)
    except Exception as e:
        print 'Could not update: %s' % e
    finally:
        connection.close()

def do_submit(host, port, task, retval):
    try:
        params = urlencode({'key_yaml': yaml.dump(task),
                            'sub_yaml': yaml.dump(retval)})
        headers = {"Content-type": "application/x-www-form-urlencoded",
                   "Accept": "text/plain"}
        connection = HTTPConnection(host, port)
        req = connection.request('POST', '/submit', params, headers)
        resp = connection.getresponse()
        if resp.status == 200:
            body = resp.read()
            return True
        else:
            raise Exception('Got code: %d' % resp.status)
    except Exception as e:
        print 'Could not submit: %s' % e
    finally:
        connection.close()

def client_loop(target, host, port=PORT, blacklist=None):
    def wrapper(task, callback):
        try:
            callback.retval = target(task, callback)
        except Exception as e:
            callback.fail = True
            traceback.print_exc()
            print 'Task Failed: %s' % e

    while True:
        print 'Getting task...'
        for attempt in count():
            task = do_request(host, port)
            if task is not None: break
            stime = min(2**attempt, 300)
            print 'No task available; trying again in %d sec.' % stime
            time.sleep(stime)

        callback = StatusCallback(str(task['key']))
        if blacklist is not None and os.path.exists(blacklist):
            print 'Checking task...'
            with open(blacklist, 'r') as f:
                for line in f:
                    pattern = line.strip()
                    if re.match(pattern, callback.task_string):
                        print ('Task "%s" matches blacklist entry "%s"'
                                % (callback.task_string, pattern))
                        callback.quit = True
                        break

        if not callback.quit:
            target_thread = Thread(target=wrapper, args=(task, callback))
            target_thread.daemon = True
            print 'Starting task...'
            target_thread.start()
            while target_thread.is_alive():
                do_update(host, port, task['key'])
                for i in range(UPDATE_INTERVAL):
                    if not target_thread.is_alive():
                        break
                    time.sleep(1)
            target_thread.join()

        if callback.quit:
            print 'Task aborted.'
            do_update(host, port, task['key'], 'quit')
        elif callback.fail or callback.retval is None:
            print 'Task failed!'
            do_update(host, port, task['key'], 'fail')
        else:
            print 'Task finished.'
            print 'Submitting result...'
            if do_submit(host, port, task['key'], callback.retval):
                print 'Result submitted!'

def test_target(task, status_callback):
    start = time.time()
    n = 5
    for i in range(n):
        print '%d: %s' % (i, task)
        time.sleep(1)
    duration = time.time() - start
    results = {}
    results['stats'] = {'time': duration}
    results['preds'] = { 123: 3.21, 456: 6.54 }
    return results

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog [options] hostname")
    parser.add_option('-b', '--blacklist-file', dest='blacklist',
                      type='str', metavar='FILE', default=None)
    options, args = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        exit()

    from experiment import client_target
    client_loop(client_target, args[0], blacklist=options.blacklist)
