import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from collections import defaultdict
import logging
import os
import platform

from operator import itemgetter

from joblib import Parallel, delayed
from iterpipes import cmd, run

from thesisgenerator.plugins.thesaurus_loader import read_thesaurus_with_caching

hostname = platform.node()
if 'apollo' in hostname or 'node' in hostname:
    orig_f = '/FeatureExtrationToolkit/feoutput-deppars/exp6-collated/exp6'
    byblo_path = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0'
else:
    # orig_f = '/Volumes/LocalDataHD/mmb28/Desktop/down/exp6-transfer'
    orig_f = '/Volumes/LocalDataHD/mmb28/Desktop/down/e6h'
    byblo_path = '/Volumes/LocalDataHD/mmb28/NetBeansProjects/Byblo-2.2.0'


def remove_punctuation(fin):
    fout = fin + '-clean'
    logging.info('Removing punctuation from %s, output will be %s' % (
        fin, fout))
    with open('words') as infile:
        # todo assume a dictionary file exists in cwd
        all_words = set(map(str.strip, infile.readlines()))
    map(str.lower, all_words)
    print len(all_words)

    with open(fin) as infile:
        with open(fout, 'w') as outfile:
            for i, line in enumerate(infile):
                things = line.split('\t')

                word = things[0].split('/')[0]
                if word.lower() not in all_words:
                    continue
                    # 33.6k --> 38.8k entry occurrences without this filter,
                # 5.6k --> 8k total entries (30% of entries across all PoS
                # not in dictionary). For comparison, 14571/76866=19% of nouns
                # in the 6-12a thesaurus are in the dict, and 3585/4103=87%
                # of verbs in 6-12b. All missing verbs are missing (as far as
                #  I can see) because of spelling error, British/American
                # spelling or PoS tagger errors

                # clean up punctuation-only features and entries
                if '/UNK' in things[0] or '/PUNCT' in things[0]:
                    # Ignore entries with unknown PoS and punctuation. This should
                    # also remove punctuation dependency features
                    continue
                things = [x for x in things if 'punct-DEP:' not in x]
                outfile.write('\t'.join(things))
    return fout


def specialise_token_occurences(fin):
    # this does not very much- 200/33k new entries appear
    fout = fin + '-split'
    logging.info('Splitting vectors from %s, output will be %s' % (
        fin, fout))
    with open(fin) as infile:
        with open(fout, 'w') as outfile:
            for i, line in enumerate(infile):
                things = line.split('\t')

                # mark verb occurrence as transitive/ intransitive
                if '/V' in things[0]:
                    if 'obj-DEP' in line:
                        # both direct and indirect object
                        things[0] = '{}/{}'.format(things[0], 'has_obj')
                    else:
                        things[0] = '{}/{}'.format(things[0], 'no_obj')

                # mark noun occurrences as subject, direct/indirect object
                # todo all other occurrences are conflated as "generic"---
                # this is not how verbs are handled
                # motivation: substitutability in context, differentiate
                # between the subject and object use of a noun: democracy can
                #  be upheld (object) but cannot uphold things
                # normal dependency thesaurus lists nouns of different degree
                #  of abstractness as neighbours
                for relation in ['dobj-HEAD:', 'iobj-HEAD:',
                                 'nsubj-HEAD:', 'nsubjpass-HEAD']: #pobj
                    if relation in line:
                        things[0] = '{}/{}'.format(things[0], relation[1:4])
                        # conflate direct/indirect object, active/passive subj

                # identify substantiated adjectives
                # if '/J' in things[0] and 'nsubj-DEP' in line \
                #     and 'det-HEAD' in line and 'amod-HEAD' not in line:
                #     print 'substantiated adjective ', i, line.strip()

                if things[0].count('/') > 2:
                    # sometimes because of parsing errors a verb is be
                    # marked as an object of another verb, ignore such weird
                    # cases
                    continue

                # todo adverbs the modify a verb/adjective
                outfile.write('\t'.join(things))
                # "I get scared" - scare is a passive verb, I is the subject

    return fout


def _do_work_byblo(features_file):
    c1 = cmd(
        './byblo.sh -s enumerate,count,filter -i {} -o {} -ffp '
        '".*(-DEP|-HEAD):''.+" -t 30',
        features_file, os.path.dirname(features_file))
    c2 = cmd('./unindex-events.sh {}', features_file)
    outfile = features_file + '.events.strings'
    # c3 = cmd('sort {} -o {}', outfile, outfile)

    for c in (c1, c2):
        out = run(c)
        print "***Byblo output***"
        print ''.join(out)
        print "***End Byblo output***"
        # ./byblo.sh -s enumerate,count,filter -i tmp/exp6-head2.txt-processed.txt -o tmp/ -ffp ".*(-DEP|-HEAD):.+"
        # ./unindex-events-unfiltered.sh tmp/exp6-head2.txt-processed.txt
    return outfile


def byblo_unindex_both(orig_file, split_file, byblo_path):
    logging.info(
        'Unindexing %s and %s with %s' % (orig_f, split_file, byblo_path))

    # need to use Byblo-2.2.0-SNAPSHOT
    os.chdir(byblo_path)
    return Parallel(n_jobs=-1)(delayed(_do_work_byblo)(f) for f in (orig_file,
                                                                    split_file))


def get_changed_entries(before, after):
    outfile = after + '-comparo'
    logging.info('Comparing feature vectors %s and %s, output will be %s' % (
        before, after, outfile
    ))

    th1 = read_thesaurus_with_caching(thesaurus_files=[before])
    th2 = read_thesaurus_with_caching(thesaurus_files=[after])
    entry_map = defaultdict(list)

    old_keys = set(th1.keys())
    for new_entry in th2.keys():
        possible_old_entry = '/'.join(new_entry.split('/')[:2])
        if possible_old_entry in old_keys:
            entry_map[possible_old_entry].append(new_entry)

    logging.debug('Before splitting:\t %d entries IT' % len(th1))
    logging.debug('After splitting:\t %d entries IT' % len(th2))

    # ignore unsplit tokens
    entry_map = {k: v for k, v in entry_map.iteritems() if len(v) > 1}

    def _sum(it):
        return sum(map(itemgetter(1), it))

    # sometimes the splitting does not cover all the uses of a word, e.g.
    # freedom/n -> freedom/n/obj, freedom/n; i.e. not all uses of freedom/n
    # are of the classes captured by specialise_token_occurences
    unsplit = 0
    with open(outfile, 'w') as outfh:
        for old_entry, new_entries in entry_map.iteritems():
            outfh.write('before:\t{} --> {}\n'.format(old_entry,
                                                      th1[old_entry]))
            old_num_features = _sum(th1[old_entry])
            new_num_features = 0
            for new_entry in new_entries:
                outfh.write('after:\t{} --> {}\n'.format(new_entry,
                                                         th2[new_entry]))
                new_num_features += _sum(th2[new_entry])
                if new_entry == old_entry:
                    unsplit += 1
                    # print new_entry, old_entry
            outfh.write('\n')

            # check that features are not lost or added in the splitting
            if old_num_features != new_num_features:
                logging.warn('Features removed for %s' % old_entry)
                # can't assert that because some features might have been removed

        logging.debug('Un-split: %d/%d matching entries' % (unsplit,
                                                            len(entry_map)))
    return outfile


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line"
                               " %(lineno)d)\t%(levelname)s : %(""message)s"
    )

    if len(sys.argv) > 1:
        orig_f = sys.argv[1]

    clean_f = remove_punctuation(orig_f)
    specialised_f = specialise_token_occurences(clean_f)
    events = byblo_unindex_both(clean_f, specialised_f, byblo_path)
    get_changed_entries(*events)