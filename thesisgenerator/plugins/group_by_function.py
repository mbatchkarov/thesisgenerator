from collections import defaultdict
import logging
import os

from joblib import Parallel, delayed, Memory
from iterpipes import cmd, run
from thesisgenerator.plugins.thesaurus_loader import read_thesaurus

orig_f = '/Volumes/LocalDataHD/mmb28/Desktop/down/e6h'
byblo_path = '/Volumes/LocalDataHD/mmb28/NetBeansProjects/Byblo-2.2.0'
memory = Memory(cachedir=os.path.dirname(orig_f), verbose=0)


@memory.cache
def remove_punctuation(fin):
    fout = fin + '-clean'
    with open(fin) as infile:
        with open(fout, 'w') as outfile:
            for i, line in enumerate(infile):
                things = line.split('\t')

                # clean up punctuation-only features and entries
                if '/UNK' in things[0] or '/PUNCT' in things[0]:
                    # Ignore entries with unknown PoS and punctuation. This should
                    # also remove punctuation dependency features
                    continue
                things = [x for x in things if 'punct-DEP:' not in x]
                outfile.write('\t'.join(things))
    return fout


@memory.cache
def specialise_token_occurences(fin):
    fout = fin + '-split'
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
                for relation in ['dobj-HEAD:', 'iobj-HEAD:',
                                 'nsubj-HEAD:', 'nsubjpass-HEAD']: #pobj
                    if relation in line:
                        things[0] = '{}/{}'.format(things[0], relation[1:4])
                        # conflate direct/indirect object, active/passive subj

                # # identify substantiated adjectives
                # if '/J' in things[0] and 'nsubj-DEP:the' in line:
                #     print 'substantiated adjective ', i, line.strip()
                #

                outfile.write('\t'.join(things))
                # "I get scared" - scare is a passive verb, I is the subject

    return fout


def _do_work(features_file):
    c1 = cmd(
        './byblo.sh -s enumerate,count,filter -i {} -o {} -ffp ".*(-DEP|-HEAD):''.+"',
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


@memory.cache
def byblo_unindex_both(orig_file, split_file, byblo_path):
    # need to use Byblo-2.2.0-SNAPSHOT
    os.chdir(byblo_path)
    return Parallel(n_jobs=-1)(delayed(_do_work)(f) for f in (orig_file,
                                                              split_file))


def get_changed_entries(before, after):
    th1 = read_thesaurus(thesaurus_files=[before])
    th2 = read_thesaurus(thesaurus_files=[after])

    entry_map = defaultdict(list)
    for old_entry in th1.keys():
        for new_entry in th2.keys():
            if new_entry.startswith(old_entry):
                entry_map[old_entry].append(new_entry)
                # ignore unsplit tokens
    entry_map = {k: v for k, v in entry_map.iteritems() if len(v) > 1}

    outfile = after + '-comparo'
    with open(outfile, 'w') as outfh:
        for old_entry, new_entries in entry_map.iteritems():
            outfh.write('before:\t{} --> {}\n'.format(old_entry,
                                                      th1[old_entry]))
            for new_entry in new_entries:
                outfh.write('after:\t{} --> {}\n'.format(new_entry,
                                                         th2[new_entry]))
            outfh.write('---------------\n')
    return outfile


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s\t%(module)s.%(funcName)s ''(line %(lineno)d)\t%(levelname)s : %(''message)s'
    )

    logging.info('started')

    clean_f = remove_punctuation(orig_f)
    specialised_f = specialise_token_occurences(clean_f)
    events = byblo_unindex_both(clean_f, specialised_f, byblo_path)
    # events = go(sys.argv[1])

    print events
    print get_changed_entries(*events)
    # with open(events) as inf:
    #     for line in inf:
    #         print line
    #         break