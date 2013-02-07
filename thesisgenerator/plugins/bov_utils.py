import csv
import glob
import os
import re
import shutil
from subprocess import CalledProcessError
import traceback
import sys
from iterpipes import cmd, run

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


def _eval_single_thesaurus(conf_file, id, t, test_data, train_data):
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
            future = executor.submit(_eval_single_thesaurus, conf_file, id, t,
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


def _infer_thesaurus_name(conf_dir, conf_file, corpus, features, fef, pos):
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

    return corpus, features, fef, pos


def _extract_thesausus_coverage_info(found_tok, found_ty, lines, log_file,
                                     repl_tok, repl_ty, th_size, total_tok,
                                     total_ty, unk_tok, unk_ty):
    for line in lines:
        # token statistics in labelled corpus
        unk_tok_this_line = int(re.findall('Unknown tokens: ([0-9]+)',
            line)[0])
        if unk_tok_this_line > 0:
            #to tell train-time messages from test-time ones
            unk_tok.append(unk_tok_this_line)
            total_tok.append(
                int(re.findall('Total tokens: ([0-9]+)', line)[0]))
            found_tok.append(
                int(re.findall('Found tokens: ([0-9]+)', line)[0]))
            repl_tok.append(
                int(re.findall('Replaced tokens: ([0-9]+)', line)[0]))

            total_ty.append(
                int(re.findall('Total types: ([0-9]+)', line)[0]))
            unk_ty.append(
                int(re.findall('Unknown types: ([0-9]+)', line)[0]))
            found_ty.append(
                int(re.findall('Found types: ([0-9]+)', line)[0]))
            repl_ty.append(
                int(re.findall('Replaced types: ([0-9]+)', line)[0]))

    # find out how large the thesaurus was from log file
    if unk_tok_this_line > 0:
        out = run(
            cmd('grep --max-count=1 "Thesaurus contains" {}', log_file))
        line = [x.strip() for x in out]
        th_size = int(re.findall("Thesaurus contains ([0-9]+)", line[0])[0])
    return th_size


def consolidate_results(conf_dir, log_dir, output_dir):
    """
    Consolidates the results of a series of experiment to ./summary.csv
    A single thesaurus must be used in each experiment
    """
    print 'Consolidating results'
    c = csv.writer(open("summary.csv", "w"))
    c.writerow(['name', 'corpus', 'features', 'pos', 'fef', 'classifier',
                'th_size', 'total_tok',
                'unknown_tok_mean', 'unknown_tok_std',
                'found_tok_mean', 'found_tok_std',
                'replaced_tok_mean', 'replaced_tok_std',
                'total_typ',
                'unknown_typ_mean', 'unknown_typ_std',
                'found_typ_mean', 'found_typ_std',
                'replaced_typ_mean', 'replaced_typ_std',
                'metric', 'score_mean', 'score_std'])
    os.chdir(conf_dir)

    experiments = glob.glob('*.conf')
    for conf_file in experiments:
        out = run(cmd('grep --max-count=1 name= {}', conf_file))
        exp_name = [x.strip() for x in out]
        exp_name = str(exp_name[0]).split('=')[1]

        # find out thesaurus information
        total_tok, unk_tok, found_tok, repl_tok, th_size = [], [], [], [], -1
        total_ty, unk_ty, found_ty, repl_ty = [], [], [], []
        corpus, pos, features, fef, sample_size = -1, -1, -1, -1, -1

        log_file = os.path.join(log_dir, '%s.log' % exp_name)
        out = run(cmd('grep Total\ types: {}', log_file))
        thesaurus_info_present = True
        try:
            info = [x.strip() for x in out]
            lines = info[0].split('\n')
        except CalledProcessError:
            # log file does not contain thesaurus replacement stats because
            # no thesaurus was used
            print 'WARNING: log file %s does not contain thesaurus ' \
                  'information' % log_file
            thesaurus_info_present = False

        if thesaurus_info_present:
            # find out how many unknown tokens, etc there were from log file
            th_size = _extract_thesausus_coverage_info(found_tok, found_ty,
                lines, log_file,
                repl_tok, repl_ty,
                th_size, total_tok,
                total_ty, unk_tok,
                unk_ty)
            # find out the name of the thesaurus(es) from the conf file
            corpus, features, fef, pos = _infer_thesaurus_name(conf_dir,
                conf_file,
                corpus, features,
                fef, pos)

        # get label names from log file
        out = run(cmd('grep Targets\ are: {}', log_file))
        info = [x.strip() for x in out]
        lines = info[0].split('\n')[0]
        targets = re.findall('Targets\ are: (.*)', lines)[0]
        import ast

        targets = ast.literal_eval(targets)

        from numpy import mean, std

        def my_mean(x):
            return mean(x) if x else -1

        def my_std(x):
            return std(x) if x else -1

        # find out the classifier score from the final csv file
        output_file = os.path.join(output_dir, '%s.out.csv' % exp_name)
        try:
            reader = csv.reader(open(output_file, 'r'))
            _ = reader.next()   # skip over header
            for row in reader:
                classifier, metric, score_my_mean, score_my_std = row
                num = re.findall('class([0-9]+)', metric)
                if num:
                    metric = re.sub('class([0-9]+)',
                        '-%s' % targets[int(num[0])], metric)

                c.writerow(
                    [exp_name, corpus, features, pos, fef, classifier,
                     int(my_mean(th_size)), int(my_mean(total_tok)),
                     int(my_mean(unk_tok)), int(my_std(unk_tok)),
                     int(my_mean(found_tok)), int(my_std(found_tok)),
                     int(my_mean(repl_tok)), int(my_std(repl_tok)),
                     int(my_mean(total_ty)),
                     int(my_mean(unk_ty)), int(my_std(unk_ty)),
                     int(my_mean(found_ty)), int(my_std(found_ty)),
                     int(my_mean(repl_ty)), int(my_std(found_ty)),
                     metric, score_my_mean, score_my_std])
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
