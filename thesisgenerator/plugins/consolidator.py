# coding=utf-8
import ast
import csv
import glob
import os
import re

import numpy
from thesisgenerator.__main__ import parse_config_file


__author__ = 'mmb28'

"""
Goes through output and log files for a given experiment and collects
interesting information. It is then passed on to a writer object,
which may write it to a csv or to a database
"""


def consolidate_results(writer, conf_dir, log_dir, output_dir,
                        unknown_pos_stats_enabled=False):
    """
    Consolidates the results of a series of experiment and passes it on to a
    writer
    A single thesaurus must be used in each experiment
    """
    print 'Consolidating results from %s' % conf_dir

    experiments = glob.glob(os.path.join(conf_dir, '*.conf'))
    unknown_pos_stats, found_pos_stats = {}, {}
    for conf_file in experiments:
        print 'Processing file %s' % conf_file

        config_obj, configspec_file = parse_config_file(conf_file)
        # with open(get_confrc(conf_file)) as infile:
        #     confrc_txt = ''.join(infile.readlines())
        # with open(conf_file) as infile:
        #     conf_txt = ''.join(infile.readlines())


        exp_name = config_obj['name']
        keep_only_IT = config_obj['tokenizer']['keep_only_IT']
        train_handler = config_obj['feature_extraction'][
            'train_token_handler'].split('.')[-1]
        decode_handler = config_obj['feature_extraction'][
            'decode_token_handler'].split('.')[-1]
        use_tfidf = config_obj['feature_extraction']['use_tfidf']

        # find out thesaurus information
        data_shape_x, data_shape_y = [], []
        log_file = os.path.join(log_dir, '%s.log' % exp_name)

        # todo this can consume loads of memory, should really scan the file a
        # line at a time
        with open(log_file) as infile:
            log_txt = ''.join(infile.readlines())

        sizes = re.findall('Data shape is (\(.*\))', log_txt)
        sizes = [ast.literal_eval(x) for x in sizes]
        # skip the information about the test set
        sizes = numpy.array(sizes)[range(0, len(sizes), 2)]
        for x in sizes:
            data_shape_x.append(x[0])
            data_shape_y.append(x[1])

        if not data_shape_x:
            print "WARNING: training data size not  present in log file %s, " \
                  "trying the other way" % log_file
            # try the other way of getting the sample size
            try:
                x = re.findall('for each sampling (\d+) documents', log_txt)
                data_shape_x.append(int(x[0]))
            except Exception:
                print "ERROR: that failed too, returning -1"
                data_shape_x.append(-1)


        # find out how many unknown tokens, etc there were from log file
        iv_it_tok, iv_oot_tok, oov_it_tok, oov_oot_tok, iv_it_ty, \
        iv_oot_ty, oov_it_ty, oov_oot_ty, total_tok, total_typ = \
            _extract_thesausus_coverage_info(log_txt)

        # find out the name of the thesaurus(es) from the conf file
        corpus, features, fef, pos = _infer_thesaurus_name(config_obj)

        def my_mean(x):
            return numpy.mean(x) if x else -1

        def my_std(x):
            return numpy.std(x) if x else -1

        s = int(my_mean(data_shape_x))
        if unknown_pos_stats_enabled:
            unknown_pos_stats[s], found_pos_stats[s] = _pos_statistics(log_file)

        # find out the classifier score from the final csv file
        output_file = os.path.join(output_dir, '%s.out.csv' % exp_name)

        import git

        git_hash = git.Repo('.').head.commit.hexsha
        import datetime

        try:
            reader = csv.reader(open(output_file, 'r'))
            _ = reader.next()   # skip over header
            for row in reader:
                classifier, metric, score_my_mean, score_my_std = row

                writer.writerow([
                    exp_name,
                    git_hash,
                    datetime.datetime.now().date().isoformat(),

                    int(my_mean(data_shape_y)), # train_voc_mean
                    int(my_std(data_shape_y)), # train_voc_std

                    # thesaurus information, if using exp0-0a naming format
                    corpus,
                    features,
                    pos,
                    fef,

                    # experiment settings
                    int(my_mean(data_shape_x)), #sample_size
                    classifier,
                    # these need to be converted to a bool and then to an int
                    #  because mysql stores booleans as a tinyint and complains
                    #  if you pass in a python boolean
                    int(keep_only_IT),
                    train_handler,
                    decode_handler,
                    int(use_tfidf),

                    # token status
                    int(total_tok),
                    int(my_mean(iv_it_tok)),
                    int(my_std(iv_it_tok)),
                    int(my_mean(iv_oot_tok)),
                    int(my_std(iv_oot_tok)),
                    int(my_mean(oov_it_tok)),
                    int(my_std(oov_it_tok)),
                    int(my_mean(oov_oot_tok)),
                    int(my_std(oov_oot_tok)),

                    # type stats
                    int(total_typ),
                    int(my_mean(iv_it_ty)),
                    int(my_std(iv_it_ty)),
                    int(my_mean(iv_oot_ty)),
                    int(my_std(iv_oot_ty)),
                    int(my_mean(oov_it_ty)),
                    int(my_std(oov_it_ty)),
                    int(my_mean(oov_oot_ty)),
                    int(my_std(oov_oot_ty)),

                    # performance
                    metric,
                    score_my_mean,
                    score_my_std])
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

    iv_it_tok = [int(x) for x in every_other(
        re.findall('IV IT tokens: ([0-9]+)', log_txt))]

    iv_oot_tok = [int(x) for x in every_other(
        re.findall('IV OOT tokens: ([0-9]+)', log_txt))]

    oov_it_tok = [int(x) for x in every_other(
        re.findall('OOV IT tokens: ([0-9]+)', log_txt))]

    oov_oot_tok = [int(x) for x in every_other(
        re.findall('OOV OOT tokens: ([0-9]+)', log_txt))]

    iv_it_ty = [int(x) for x in every_other(
        re.findall('IV IT types: ([0-9]+)', log_txt))]

    iv_oot_ty = [int(x) for x in every_other(
        re.findall('IV OOT types: ([0-9]+)', log_txt))]

    oov_it_ty = [int(x) for x in every_other(
        re.findall('OOV IT types: ([0-9]+)', log_txt))]

    oov_oot_ty = [int(x) for x in every_other(
        re.findall('OOV OOT types: ([0-9]+)', log_txt))]

    try:
        # the pointwise sum of the token/type lists below should be a constant
        # for a given data set, what changes is not the number of
        # tokens/types, but how that distribute in the IV/OOV, IT/OOT bins
        total_tok = iv_it_tok[0] + iv_oot_tok[0] + oov_it_tok[0] + oov_oot_tok[
            0]
        total_typ = iv_it_ty[0] + iv_oot_ty[0] + oov_it_ty[0] + oov_oot_ty[0]
    except IndexError:
        total_tok, total_typ = -1, -1

    return iv_it_tok, iv_oot_tok, oov_it_tok, oov_oot_tok, iv_it_ty, \
           iv_oot_ty, oov_it_ty, oov_oot_ty, total_tok, total_typ


def _infer_thesaurus_name(config_obj):
    thesauri = config_obj['feature_extraction']['thesaurus_files']

    corpus, features, pos, fef = [], [], [], []
    if thesauri:
        for t in thesauri:
            pattern = '.*exp(?P<corpus>\d+)-(?P<features>\d+)(?P<pos>[A-Za-z])' \
                      '(fef(?P<fef>\d+))?.*'
            m = re.match(pattern, t)
            corpus.append(m.group('corpus'))
            features.append(m.group('features'))
            pos.append(m.group('pos'))
            fef.append(m.group('fef') if m.group('fef') else '?')

    return '_'.join(corpus), \
           '_'.join(features), \
           '_'.join(fef), \
           '_'.join(pos)


def _pos_statistics(input_file):
    # todo this needs to be updated to IV/IT notation
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