import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from thesisgenerator.plugins.thesaurus_loader import Thesaurus
from collections import defaultdict
import logging
import os
import platform
from glob import glob
from operator import itemgetter

from joblib import Parallel, delayed, Memory
from iterpipes import cmd, run

mem = Memory('.', verbose=0)


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


def preserve_cwd(function):
    def decorator(*args, **kwargs):
        cwd = os.getcwd()
        try:
            return function(*args, **kwargs)
        finally:
            os.chdir(cwd)

    return decorator


@preserve_cwd
@mem.cache
def specialise_token_occurences(fin):
    """
    Buckets features of each token occurence differently depending on the grammatical function of that occurrence.
    Takes a features file (output of FET, input to Byblo) and input, i.e. slots between FET and Byblo
    """

    # this does not very much- 200/33k new entries appear
    fout = fin + '-split'
    logging.info('Splitting vectors from %s, output will be %s' % (fin, fout))
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

                # identify nouns being used as adjectives
                if 'nn-HEAD' in line:
                    things[0] = '{}/{}'.format(things[0], 'nn')

                # identify substantiated adjectives
                # if '/J' in things[0] and 'nsubj-DEP' in line \
                #     and 'det-HEAD' in line and 'amod-HEAD' not in line:
                #     print 'substantiated adjective ', i, line.strip()

                if things[0].count('/') > 2:
                    # sometimes because of parsing errors a verb can be
                    # marked as an object of another verb, ignore such weird
                    # cases
                    continue

                # todo adverbs the modify a verb/adjective
                outfile.write('\t'.join(things))
                # "I get scared" - scare is a passive verb, I is the subject

    return fout


@preserve_cwd
@mem.cache
def byblo_full_build_4_pos(input_file, nthreads, byblo_path):
    """
    Builds 4 thesauri, one for each main PoS
    """

    os.chdir(byblo_path)
    pos = ['N', 'V', 'J', 'RB']
    entry_patterns = ['.+/%s.*' % x for x in pos]
    feature_patterns = [
        '(amod-DEP|dobj-HEAD|conj-DEP|conj-HEAD|iobj-HEAD|nsubj-HEAD|nn-DEP|nn-HEAD|pobj-HEAD):.+',
        '(conj-DEP|conj-HEAD|dobj-DEP|iobj-DEP|nsubj-DEP):.+',
        '(conj-DEP|conj-HEAD|amod-HEAD):.+',
        '(advmod-HEAD|conj-DEP|conj-HEAD):.+'
    ]

    input_file_name = os.path.basename(input_file)
    output_dirs = ['%s-%sthes' % (input_file, x) for x in pos]
    for d in output_dirs:
        if not os.path.exists(d):
            os.mkdir(d)
    entry_freq, feature_freq, event_freq = 1, 1, 1 # todo hardcoded filtering values for now

    for entry_pattern, feature_pattern, output_dir in zip(entry_patterns, feature_patterns, output_dirs):
        command_str = """
        ./byblo.sh --input {input_file} --output {output_dir}
        --threads {nthreads} --filter-entry-freq {entry_freq}
        --filter-feature-freq {feature_freq} --filter-event-freq {event_freq}
        --similarity-min 0.01 -k 100 --filter-entry-pattern "{entry_pattern}"
        --filter-feature-pattern "{feature_pattern}"
        """.format(**locals())

        def run_byblo_with_iterpipes(command_str):
            command_str = ' '.join(command_str.split())
            logging.info(command_str)
            c = cmd(command_str)
            out = run(c)
            logging.info("***Byblo output***")
            logging.info(''.join(out))
            logging.info("***End Byblo output***")

        run_byblo_with_iterpipes(command_str)
        command_str = """
        ./tools.sh unindex-events -i {0}.events -o {0}.events.strings
         -Xe {0}.entry-index -Xf {0}.feature-index -et JDBM
        """.format(os.path.join(output_dir, os.path.basename(input_file)))

        run_byblo_with_iterpipes(command_str)

    return output_dirs


def _byblo_enum_count_filter(features_file):
    c1 = cmd(
        './byblo.sh -s enumerate,count,filter -i {} -o {} -ffp '
        '".*(-DEP|-HEAD):''.+" -t 30',
        features_file, os.path.dirname(features_file))
    c2 = cmd('./unindex-events.sh {}', features_file)
    outfile = features_file + '.events.strings'

    for c in (c1, c2):
        out = run(c)
        logging.info("***Byblo output***")
        logging.info(''.join(out))
        logging.info("***End Byblo output***")
        # ./byblo.sh -s enumerate,count,filter -i tmp/exp6-head2.txt-processed.txt -o tmp/ -ffp ".*(-DEP|-HEAD):.+"
        # ./unindex-events-unfiltered.sh tmp/exp6-head2.txt-processed.txt
    return outfile


@mem.cache
@preserve_cwd
def byblo_unindex_both(orig_file, split_file, byblo_path):
    """
    Enumerates entries, counts events, filters entries, features and events using Byblo.
    Returns the path to the *.events.strings files corresponding to the first two input parameters
    """
    logging.info('Unindexing %s and %s with %s' % (orig_f, split_file, byblo_path))

    # need to use Byblo-2.2.0-SNAPSHOT
    os.chdir(byblo_path)
    return Parallel(n_jobs=-1)(delayed(_byblo_enum_count_filter)(f) for f in (orig_file, split_file))


@mem.cache
def get_changed_feature_vectors(events_before, events_after):
    """
    Finds entries whose feature vector has been modified by specialise_token_occurences, and prints these to a file
    for inspection.

    """
    outfile = events_after + '-comparo.strings'
    logging.info('Comparing feature vectors %s and %s, output will be %s' % (events_before, events_after, outfile))

    # using the Thesaurus class to read Byblo events files because these have the same format as thesaurus files
    th1 = Thesaurus(thesaurus_files=[events_before])
    th2 = Thesaurus(thesaurus_files=[events_after])
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
            outfh.write('before:\t{} --> {}\n'.format(old_entry, th1[old_entry]))
            old_num_features = _sum(th1[old_entry])
            new_num_features = 0
            for new_entry in new_entries:
                outfh.write('after:\t{} --> {}\n'.format(new_entry, th2[new_entry]))
                new_num_features += _sum(th2[new_entry])
                if new_entry == old_entry:
                    unsplit += 1
                    # print new_entry, old_entry

            # check that features are not lost or added in the splitting
            if old_num_features != new_num_features:
                logging.warn('{} features removed for {}'.format(old_num_features - new_num_features, old_entry))
                # can't assert that because some features might have been removed
                outfh.write('{} features removed \n'.format(old_num_features - new_num_features))
            outfh.write('\n')

        logging.debug(
            '%d/%d  entries have a "generic" feature vector that has not been split according to function' % (
                unsplit, len(entry_map)))
    return outfile, entry_map


def get_changed_neighbours(changed_entries, old_thes_file, new_thes_file):
    outfile = new_thes_file + '.comparo.txt'

    logging.info('Comparing entries in thesauri {} and {}, output will be {}'.format(old_thes_file,
                                                                                     new_thes_file,
                                                                                     outfile))

    th1 = Thesaurus(thesaurus_files=[old_thes_file])
    th2 = Thesaurus(thesaurus_files=[new_thes_file])

    old_entries_with_this_pos = set(th1.keys())
    changed_entries_this_pos = {x: y for (x, y) in changed_entries.iteritems() if x in old_entries_with_this_pos}

    with open(outfile, 'w') as outfh:
        for old_entry, new_entries in changed_entries_this_pos.iteritems():
            outfh.write('before:\t{} --> {}\n'.format(old_entry, th1[old_entry]))
            for new_entry in new_entries:
                try:
                    outfh.write('after:\t{} --> {}\n'.format(new_entry, th2[new_entry]))
                except KeyError:
                    outfh.write('after:\t{} has been filtered out -------------------------\n'.format(new_entry))
                    # because it has very few or the wrong features, because it does not have any neighbours with
                    # sim >0.01, etc...
                    logging.warn('{} not in thesaurus'.format(new_entry))
            outfh.write('\n')
    return outfile


if __name__ == '__main__':
    hostname = platform.node()
    if 'apollo' in hostname or 'node' in hostname:
        orig_f = '/FeatureExtrationToolkit/feoutput-deppars/exp6-collated/exp6'
        byblo_path = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/Byblo-2.2.0'
    else:
        # orig_f = '/Volumes/LocalDataHD/mmb28/Desktop/down/exp6-transfer'
        orig_f = '/Volumes/LocalDataHD/mmb28/Desktop/down/e6h'
        byblo_path = '/Volumes/LocalDataHD/mmb28/NetBeansProjects/Byblo-2.2.0'

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(""message)s")

    if len(sys.argv) > 1:
        orig_f = sys.argv[1]

    clean_f = orig_f
    specialised_f = specialise_token_occurences(clean_f)

    # thesaurus_dirs_old = byblo_full_build_4_pos(clean_f, 3, byblo_path)
    # thesaurus_dirs_new = byblo_full_build_4_pos(specialised_f, 3, byblo_path)
    thesaurus_dirs_old = [x for x in glob('%s*thes*' % (orig_f)) if 'split' not in x]
    thesaurus_dirs_new = [x for x in glob('%s*thes*' % (orig_f)) if 'split' in x]

    for old_thes_dir, new_thes_dir in zip(thesaurus_dirs_old, thesaurus_dirs_new):
        def get_thes_file(input_file, thes_dir):
            return os.path.join(
                thes_dir,
                '{}.sims.neighbours.strings'.format(os.path.basename(input_file))
            )

        def get_events_file(input_file, thes_dir):
            return os.path.join(
                thes_dir,
                '{}.events.strings'.format(os.path.basename(input_file))
            )

        outfile, changed_entries = get_changed_feature_vectors(get_events_file(clean_f, old_thes_dir),
                                                               get_events_file(specialised_f, new_thes_dir))

        # logging.debug('Bucketed entries are:')
        # logging.debug(pformat(changed_entries))
        if changed_entries:
            get_changed_neighbours(changed_entries,
                                   get_thes_file(clean_f, old_thes_dir),
                                   get_thes_file(specialised_f, new_thes_dir)
            )
            # break