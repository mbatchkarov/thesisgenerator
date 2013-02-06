import csv
import glob
import os
import re
import shutil
from subprocess import CalledProcessError
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
    print 'Classifying with thesauri %s' % thesauri
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


def consolidate_results(conf_dir, log_dir, output_dir):
    """
    Consolidates the results of a series of experiment to ./summary.csv
    A single thesaurus must be used in each experiment
    """
    print 'Consolidating results'
    c = csv.writer(open("summary.csv", "w"))
    c.writerow(['ID', 'corpus', 'features', 'pos', 'fef', 'classifier',
                'th_size', 'total_tok', 'unknown_tok', 'found_tok',
                'replaced_tok', 'total_typ', 'unknown_typ', 'found_typ',
                'replaced_typ', 'metric', 'score'])

    os.chdir(conf_dir)
    from iterpipes import cmd, run

    experiments = glob.glob('*.conf')
    for conf_file in experiments:
        out = run(cmd('grep --max-count=1 name= {}', conf_file))
        exp_name = [x.strip() for x in out]
        exp_name = str(exp_name[0]).split('=')[1]

        log_file = os.path.join(log_dir, '%s.log' % exp_name)
        out = run(cmd('grep --max-count=2 Total\ types: {}', log_file))
        try:
            info = [x.strip() for x in out]
            info = info[0].split('\n')[1]

            # token statistics in labelled corpus
            total_tok = int(re.findall('Total tokens: ([0-9]+)', info)[0])
            unk_tok = int(re.findall('Unknown tokens: ([0-9]+)', info)[0])
            found_tok = int(re.findall('Found tokens: ([0-9]+)', info)[0])
            repl_tok = int(re.findall('Replaced tokens: ([0-9]+)', info)[0])

            total_ty = int(re.findall('Total types: ([0-9]+)', info)[0])
            unk_ty = int(re.findall('Unknown types: ([0-9]+)', info)[0])
            found_ty = int(re.findall('Found types: ([0-9]+)', info)[0])
            repl_ty = int(re.findall('Replaced types: ([0-9]+)', info)[0])

            out = run(
                cmd('grep --max-count=1 "Thesaurus contains" {}', log_file))
            info = [x.strip() for x in out]
            th_size = re.findall("Thesaurus contains ([0-9]+)", info[0])[0]

            abs_conf_file = os.path.join(conf_dir, '%s' % conf_file)
            out = run(
                cmd('grep --max-count=1 thesaurus_files {}', abs_conf_file))
            out = [x.strip() for x in out]
            thesauri = out[0].split('=')[1]
            thesauri = os.sep.join(thesauri.split(os.sep)[-2:])
            # thesauri is something like "exp6-11a/exp6.sims.neighbours.strings,"
            if thesauri:
                corpus = re.findall('exp([0-9]+)', thesauri)[0]
                features = re.findall('-([0-9]+)', thesauri)[0]
                pos = re.findall('-[0-9]+(.)\..*', thesauri)[0]
                try:
                    fef = re.findall('fef([0-9]+)', thesauri)[0]
                except IndexError:
                # 'fef' isn't in thesaurus name, i.e. has not been postfiltered
                    fef = 0
                    print 'WARNING: thesaurus file name %s does not contain ' \
                          'explicit fef information' % thesauri
            else:
                # a thesaurus was not used
                corpus, features, pos, fef = -1, -1, -1, -1
        except CalledProcessError:
            # log file does not contain thesaurus replacement stats because
            # no thesaurus was used
            print 'WARNING: log file %s does not contain thesaurus ' \
                  'information' % log_file
            total_tok, unk_tok, found_tok, repl_tok, th_size = -1, -1, -1, -1, -1
            total_ty, unk_ty, found_ty, repl_ty = -1, -1, -1, -1
            corpus, pos, features, fef = -1, -1, -1, -1

        output_file = os.path.join(output_dir, '%s.out.csv' % exp_name)
        try:
            reader = csv.reader(open(output_file, 'r'))
            _ = reader.next() # skip over header
            for row in reader:
                _, classifier, metric, score = row
                c.writerow([exp_name, corpus, features, pos, fef, classifier,
                            th_size, total_tok, unk_tok, found_tok, repl_tok,
                            total_ty, unk_ty, found_ty, repl_ty, metric,
                            score])
        except IOError:
            print 'WARNING: %s is missing' % output_file
            continue    # file is missing


if __name__ == '__main__':
    # on local machine
    # evaluate_thesauri(
    #     '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/Byblo-2.1.0/exp6-11*/*sims.neighbours.strings',
    #     '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/conf/main.conf',
    #     pool_size=10)

    # on cluster
    # evaluate_thesauri(
    #     '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/exp6-*/*sims.neighbours.strings',
    #     '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/conf/main.conf',
    #     train_data='/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/reuters21578/r8train-tagged',
    #     test_data='/mnt/lustre/scratch/inf/mmb28/thesisgenerator/sample-data/reuters21578/r8test-tagged',
    #     pool_size=30)

    # on local machine
    consolidate_results(
        '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/conf/',
        '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/conf/logs/',
        '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/conf/output/',
    )

    # on cluster
    # consolidate_results(
    #     '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/conf/main-variants/',
    #     '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/conf/main-variants/logs/',
    #     '/mnt/lustre/scratch/inf/mmb28/thesisgenerator/conf/output/',
    # )
