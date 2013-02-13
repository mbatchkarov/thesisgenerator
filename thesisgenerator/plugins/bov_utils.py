import csv
import glob
from itertools import chain
import os
import re
import shutil
from subprocess import CalledProcessError
import sys
from iterpipes import cmd, run

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


def _write_exp1_conf_file(base_conf_file, run_id, thes):
    """
    Create a bunch of copies of the base conf file and in the new one modifies
    the experiment name and the (SINGLE) thesaurus used
    """
    log_file, new_conf_file = _prepare_conf_files(base_conf_file, 1,
        run_id)

    replace_in_file(new_conf_file, 'name=.*', 'name=exp%d-%d' % (1, run_id))

    # it is important that the list of thesaurus files in the conf file ends with a comma
    replace_in_file(new_conf_file, 'thesaurus_files=.*',
        'thesaurus_files=%s,' % thes)
    return new_conf_file, log_file


def _exp1_file_iterator(pattern, base_conf_file):
    """
    Generates a conf file for each thesaurus file matching the provided glob
    pattern. Each conf file is a single classification run
    """
    thesauri = glob.glob(pattern)

    for id, t in enumerate(thesauri):
        new_conf_file, log_file = _write_exp1_conf_file(base_conf_file, id, t)
        yield new_conf_file, log_file
    raise StopIteration


def _prepare_conf_files(base_conf_file, exp_id, run_id):
    """
    Takes a conf files and moves it to a subdirectory to be modified later.
    Also creates a log file for the classification run that will be run with
    the newly created conf file
     Parameter id identifies the clone and distinguishes it from other clones
    """
    name, ext = os.path.splitext(base_conf_file)
    name = '%s-variants' % name
    if not os.path.exists(name):
        os.mkdir(name)
    new_conf_file = os.path.join(name, 'exp%d-%d%s' % (exp_id, run_id, ext))
    log_file = os.path.join(name, '..', 'logs')
    shutil.copy(base_conf_file, new_conf_file)
    configspec_file = os.path.join(os.path.dirname(base_conf_file), '.confrc')
    shutil.copy(configspec_file, name)
    return log_file, new_conf_file


def _write_exp2_3_conf_file(base_conf_file, run_id, sample_size):
    """
    Prepares conf files for exp2 by altering the sample size parameter in the
     provided base conf file
    """
    log_file, new_conf_file = _prepare_conf_files(base_conf_file, 2, run_id)

    replace_in_file(new_conf_file, 'name=.*',
        'name=exp%d-%d' % (2, run_id))
    replace_in_file(new_conf_file, 'sample_size=.*',
        'sample_size=%s' % sample_size)
    return new_conf_file, log_file


def _exp2and3_file_iterator(sizes, conf_file):
    for id, size in enumerate(sizes):
        new_conf_file, log_file = _write_exp2_3_conf_file(conf_file, id,
            size)
        yield new_conf_file, log_file
    raise StopIteration


def evaluate_thesauri(file_iterator, pool_size=1):
    from joblib import Parallel, delayed

    Parallel(n_jobs=pool_size)(delayed(go)(new_conf_file, log_file) for
        new_conf_file, log_file in file_iterator)


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
        pos = re.findall('-[0-9]+(.)', thesauri)[0]
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
    os.chdir(conf_dir)
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
    prefix = '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator'
    pattern = '%s/../Byblo-2.1.0/exp6-1*/*sims.neighbours.strings' % (prefix)
    num_workers = 4

    # on cluster
    # prefix = '/mnt/lustre/scratch/inf/mmb28/thesisgenerator'
    # pattern = '%s/../FeatureExtrationToolkit/exp6-1*/*sims.neighbours' \
    #           '.strings' % prefix
    # num_workers = 30

    # ----------- EXPERIMENT 1 -----------
    # it1 = _exp1_file_iterator(pattern, '%s/conf/exp1/exp1_base.conf' % prefix)
    # evaluate_thesauri(it1, pool_size=num_workers)

    # ----------- EXPERIMENT 2 -----------
    sizes = chain([50], range(100, 1001, 100), range(1500, 500, 500))
    # last value is the total number of documents
    it2 = _exp2and3_file_iterator(sizes,
        '%s/conf/exp2/exp2_base.conf' % (prefix)
    )
    evaluate_thesauri(it2, pool_size=num_workers)

    # ----------- CONSOLIDATION -----------
    for i in range(1, 3):
        consolidate_results(
            '%s/conf/exp%d/exp%d_base-variants' % (prefix, i, i),
            '%s/conf/exp%d/logs/' % (prefix, i),
            '%s/conf/exp%d/output/' % (prefix, i),
        )