import ast
import csv
import glob
import os
import re
import numpy

__author__ = 'mmb28'

"""
Goes through output and log files for a given experiment and collects
interesting information. It is then passed on to a writer object,
which may write it to a csv or to a sqlite database
"""


def consolidate_results(writer, conf_dir, log_dir, output_dir,
                        unknown_pos_stats_enabled=False):
    """
    Consolidates the results of a series of experiment and passes it on to a
    writer
    A single thesaurus must be used in each experiment
    """
    print 'Consolidating results from %s' % conf_dir
    os.chdir(conf_dir)

    experiments = glob.glob('*.conf')
    unknown_pos_stats, found_pos_stats = {}, {}
    for conf_file in experiments:
        print 'Processing file %s' % conf_file
        with open(conf_file) as infile:
            conf_txt = ''.join(infile.readlines())
        exp_name = re.findall('name=(.*)', conf_txt)[0]

        # find out thesaurus information
        data_shape_x, data_shape_y = [], []
        log_file = os.path.join(log_dir, '%s.log' % exp_name)

        with open(log_file) as infile:
            log_txt = ''.join(infile.readlines())

        lines = re.findall('Total types:', log_txt)
        if not lines:
            print 'WARNING: log file %s does not contain thesaurus ' \
                  'information' % log_file

        sizes = re.findall('Data shape is (\(.*\))', log_txt)
        sizes = [ast.literal_eval(x) for x in sizes]
        # skip the information about the test set
        sizes = numpy.array(sizes)[range(0, len(sizes), 2)]
        for x in sizes:
            data_shape_x.append(x[0])
            data_shape_y.append(x[1])

        if not data_shape_x:
            print "WARNING: training data size not  present in log file %s, " \
                  "trying the other way" \
                  "" % \
                  log_file
            # try the other way of getting the sample size
            try:
                x = re.findall('for each sampling (\d+) documents', log_txt)
                data_shape_x.append(int(x[0]))
            except Exception:
                print "ERROR: that failed too, returning -1"
                data_shape_x.append(-1)


        # find out how many unknown tokens, etc there were from log file
        total_tok, total_ty, unk_tok, unk_ty, found_tok, found_ty, repl_tok, \
        repl_ty, th_size = _extract_thesausus_coverage_info(log_txt)

        # find out the name of the thesaurus(es) from the conf file
        corpus, features, fef, pos = _infer_thesaurus_name(conf_txt)

        def my_mean(x):
            return numpy.mean(x) if x else -1

        def my_std(x):
            return numpy.std(x) if x else -1

        s = int(my_mean(data_shape_x))
        if unknown_pos_stats_enabled:
            unknown_pos_stats[s], found_pos_stats[s] = _pos_statistics(log_file)

        # find out the classifier score from the final csv file
        output_file = os.path.join(output_dir, '%s.out.csv' % exp_name)
        try:
            reader = csv.reader(open(output_file, 'r'))
            _ = reader.next()   # skip over header
            for row in reader:
                classifier, metric, score_my_mean, score_my_std = row

                writer.writerow(
                    [exp_name, int(my_mean(data_shape_x)),
                     int(my_mean(data_shape_y)), int(my_std(data_shape_y)),
                     corpus, features, pos, fef, classifier,
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

    if unknown_pos_stats:
        from pandas import DataFrame

        df = DataFrame(unknown_pos_stats).T
        df.to_csv('unknown_token_stats.csv')
        df = DataFrame(found_pos_stats).T
        df.to_csv('found_token_stats.csv')


def _extract_thesausus_coverage_info(log_txt):
    # token statistics in labelled corpus

    def every_other(iterable):
        """Returns every other element in a iterable in a silly way"""
        return numpy.array(iterable)[range(1, len(iterable), 2)]

    unk_tok = [int(x) for x in every_other(
        re.findall('Unknown tokens: ([0-9]+)', log_txt))]

    total_tok = [int(x) for x in every_other(
        re.findall('Total tokens: ([0-9]+)', log_txt))]
    found_tok = [int(x) for x in every_other(
        re.findall('Found tokens: ([0-9]+)', log_txt))]

    repl_tok = [int(x) for x in every_other(
        re.findall('Replaced tokens: ([0-9]+)', log_txt))]

    total_ty = [int(x) for x in every_other(
        re.findall('Total types: ([0-9]+)', log_txt))]

    unk_ty = [int(x) for x in every_other(
        re.findall('Unknown types: ([0-9]+)', log_txt))]
    found_ty = [int(x) for x in every_other(
        re.findall('Found types: ([0-9]+)', log_txt))]
    repl_ty = [int(x) for x in every_other(
        re.findall('Replaced types: ([0-9]+)', log_txt))]

    # find out how large the thesaurus was from log file
    th_size = re.findall("Thesaurus contains ([0-9]+)", log_txt)
    th_size = int(th_size[0]) if th_size else -1

    return total_tok, total_ty, unk_tok, unk_ty, found_tok, found_ty, \
           repl_tok, repl_ty, th_size


def _infer_thesaurus_name(conf_txt):
    thesauri = ''.join(re.findall('thesaurus_files\s*=([\w-]+)', conf_txt))
    if thesauri:
    # thesauri is something like "exp6-11a/exp6.sims.neighbours.strings,"
        corpus = re.findall('exp([0-9]+)', thesauri)[0]
        features = (re.findall('-([0-9]+)', thesauri))[0]
        pos = (re.findall('-[0-9]+(.)', thesauri))[0]
        fef = re.findall('fef([0-9]+)', thesauri)
        # 'fef' isn't in thesaurus name, i.e. has not been postfiltered
        if not fef:
            fef = 0
            print 'WARNING: thesaurus file name %s does not contain ' \
                  'explicit fef information' % thesauri
    else:
        # a thesaurus was not used
        corpus, features, pos, fef = -1, -1, -1, -1

    return corpus, features, fef, pos


def _pos_statistics(input_file):
    regex1 = re.compile(".*Unknown token.*/(.*)")
    regex2 = re.compile(".*Found thesaurus entry.*/(.*)")
    unknown_pos, found_pos = [], []
    with open(input_file) as infile:
        for line in infile:
            matches = regex1.findall(line)
            if matches:
                unknown_pos.append(matches[0])

            matches = regex2.findall(line)
            if matches:
                found_pos.append(matches[0])

    from collections import Counter

    return Counter(unknown_pos), Counter(found_pos)