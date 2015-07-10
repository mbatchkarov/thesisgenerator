import logging
import os
import pwd
import subprocess as sub
from discoutils.cmd_utils import run_and_log_output

__author__ = 'mmb28'

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s",
                    datefmt='%m-%d %H:%M')


def _get_user_name():
    return pwd.getpwuid(os.getuid()).pw_name


with open('slow_experiments.txt') as infile:
    slow = set(int(x.strip()) for x in infile.readlines() if x.strip())

qs_str = sub.check_output('qstat -u {}'.format(_get_user_name()), shell=True)
logging.info(qs_str)
main_job_id = int(input('What is the ID of the job array. See above for your jobs? >>> '))

logging.info('Will delete %d sub-jobs', len(slow))
for eid in sorted(list(slow)):
    run_and_log_output('qdel %d.%d' % (main_job_id, eid))
