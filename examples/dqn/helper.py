import pytz
import subprocess
import sys
import time
import torch

from datetime import datetime
from os.path import dirname, realpath

try:
    term_width = int(subprocess.check_output('stty size'.split()).decode('UTF-8').split()[0])
except:
    print('stty command failed with error ({})'.format(sys.exc_info()[0]))
    term_width = 171

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def vec_stats(vec):
    return [func(vec).item() for func in [torch.mean, torch.median, torch.std, torch.min, torch.max]]

def format_time(f):
    return datetime.fromtimestamp(f, tz=pytz.utc).strftime('%H:%M:%S.%f s')

def percent_time(total_time, *times):
    return [(t / total_time) * 100 for t in times]

def progress_bar(current, total, msg=None):
    global last_time, begin_time

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for _ in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for _ in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for _ in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for _ in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n\n')

    sys.stdout.flush()

