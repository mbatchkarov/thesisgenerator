import csv
import glob
import os
import re
import shutil
import traceback
import sys

from concurrent.futures import as_completed
from numpy import nonzero


try:
    from thesisgenerator.__main__ import go
    from thesisgenerator.utils import replace_in_file
except ImportError:
# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
    sys.path.append('../../')
    sys.path.append('../')
    sys.path.append('./')
    sys.path.append('./thesisgenerator')
    from thesisgenerator.__main__ import go
    from thesisgenerator.utils import replace_in_file

__author__ = 'mmb28'


def inspect_thesaurus_effect(outdir, clf_name, thesaurus_file, pipeline,
                             predicted, x_test):
    """
    Evaluates the performance of a classifier with and without the thesaurus
    that backs its vectorizer
    """

    # remove the thesaurus
    pipeline.named_steps['vect'].thesaurus = {}
    predicted2 = pipeline.predict(x_test)

    with open('%s/before_after_%s.csv' % (outdir, thesaurus_file),
              'a') as outfile:
        outfile.write('DocID,')
        outfile.write(','.join([str(x) for x in range(len(predicted))]))
        outfile.write('\n')
        outfile.write('%s+Thesaurus,' % clf_name)
        outfile.write(','.join([str(x) for x in predicted.tolist()]))
        outfile.write('\n')
        outfile.write('%s-Thesaurus,' % clf_name)
        outfile.write(','.join([str(x) for x in predicted2.tolist()]))
        outfile.write('\n')
        outfile.write('Decisions changed: %d' % (
            nonzero(predicted - predicted2)[0].shape[0]))
        outfile.write('\n')


def _do_single_thesaurus(conf_file, id, t, test_data, train_data):
    name, ext = os.path.splitext(conf_file)
    name = '%s-variants' % name
    if not os.path.exists(name):
        os.mkdir(name)
    new_conf_file = os.path.join(name, 'run%d%s' % (id, ext))
    log_file = os.path.join(name, 'logs')
    shutil.copy(conf_file, new_conf_file)
    configspec_file = os.path.join(os.path.dirname(conf_file), '.confrc')
    shutil.copy(configspec_file, name)

    replace_in_file(new_conf_file, 'name=.*', 'name=run%d' % id)
    replace_in_file(new_conf_file, 'training_data=.*',
                    'training_data=%s' % train_data)
    replace_in_file(new_conf_file, 'test_data=.*',
                    'test_data=%s' % test_data)
    # it is important that the list of thesaurus files in the conf file ends with a comma
    replace_in_file(new_conf_file, 'thesaurus_files=.*',
                    'thesaurus_files=%s,' % t)
    return go(new_conf_file, log_file)


def evaluate_thesauri(pattern, conf_file, train_data='sample-data/web-tagged',
                      test_data='sample-data/web2-tagged', pool_size=1):
    from concurrent.futures import ProcessPoolExecutor
    thesauri = glob.glob(pattern)
    print 'Classifying with thesauri %s'%thesauri
    #Create a number (processes) of individual processes for executing parsers.
    with ProcessPoolExecutor(max_workers=pool_size) as executor:
        jobs = {} #Keep record of jobs and their input
        id = 1
        for t in thesauri:
            future = executor.submit(_do_single_thesaurus, conf_file, id, t,
                                     test_data, train_data)
            jobs[future] = id
            id += 1

        #As each job completes, check for success, print details of input
        for job in as_completed(jobs.keys()):
            try:
                status, outfile = job.result()
                print("Success. Files produced: %s" % outfile)
            except Exception as exc:
                # print(
                #     " Exception encountered in: \n-- %s" % "\n-- ".join(
                #         jobs[job]))
                print(''.join(traceback.format_exception(*sys.exc_info())))
                raise exc


def consolidate_results(log_dir, output_dir):
    """

    A single thesaurus must be used in each experiment
    """
    print 'Consolidating results'
    c = csv.writer(open("summary.csv", "w"))
    c.writerow(['ID', 'corpus', 'features', 'pos', 'fef', 'classifier', 'th_size', 'vocab_size', 'unknown', 'replaced',
                'accuracy'])

    os.chdir(log_dir)
    from iterpipes import cmd, run

    experiments = glob.glob('*.conf')
    for exp in experiments:
        id = re.findall(r'[a-zA-Z]*([0-9]+).conf', exp)[0]
        log_file = os.path.join(log_dir, 'logs', 'run%s.log' % id)
        out = run(cmd('grep --max-count=2 Total {}', log_file))
        info = [x.strip() for x in out]
        info = info[0].split('\n')[1]

        # token statistics in labelled corpus
        total = int(re.findall('Total: ([0-9]+)', info)[0])
        unk = int(re.findall('Unknown: ([0-9]+)', info)[0])
        repl = int(re.findall('Replaced: ([0-9]+)', info)[0])
        # print id, total, unk, repl

        out = run(cmd('grep --max-count=1 "Thesaurus contains" {}', log_file))
        info = [x.strip() for x in out]
        th_size = re.findall("Thesaurus contains ([0-9]+)", info[0])[0]

        conf_file = os.path.join(log_dir, 'run%d.conf' % int(id))
        out = run(cmd('grep --max-count=1 thesaurus_files {}', conf_file))
        out = [x.strip() for x in out]
        thesauri = out[0].split('=')[1]
        thesauri = os.sep.join(thesauri.split(os.sep)[-2:])
        # thesauri is something like "exp6-11a/exp6.sims.neighbours.strings,"

        corpus = re.findall('exp([0-9]+)', thesauri)[0]
        features = re.findall('-([0-9]+)', thesauri)[0]
        pos = re.findall('-[0-9]+(.*)/', thesauri)[0]
        try:
            fef = re.findall('fef([0-9]+)', thesauri)[0]
        except IndexError:
            # 'fef' isn't in thesaurus name, i.e. has not been postfiltered
            fef = 0


        output_file = os.path.join(output_dir, 'run%d.out.csv' % int(id))
        try:
            reader = csv.reader(open(output_file, 'r'))
            header = reader.next()
            for row in reader:
                _, classifier, metric, score = row
                c.writerow([id, corpus, features, pos, fef, classifier, th_size, total, unk, repl, score])
        except IOError:
            continue #file is missing

if __name__ == '__main__':

    # on local machine
    # evaluate_thesauri(
    #     '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/Byblo-2.1.0/exp6-11*/*sims.neighbours.strings',
    #     '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/conf/main.conf',
    #     pool_size=10)

    # on cluster
    evaluate_thesauri(
        '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-*/*sims.neighbours.strings',
        '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/conf/main.conf',
        train_data='/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/reuters21578/r8train-tagged',
        test_data='/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/reuters21578/r8test-tagged',
        pool_size=30)

    # on local machine
    # consolidate_results(
    #     '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/conf/main-variants/',
    #     '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/conf/output/',
    # )

    # on cluster
    consolidate_results(
        '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/conf/main-variants/',
        '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/conf/output/',
    )
