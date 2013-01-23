import glob
from operator import itemgetter
import os
import random
import numpy
from scipy.stats import kendalltau

__author__ = 'mmb28'


def compare_thesauri(prefix, names):
    """
    Selects random 50 in-vocabulary entries which occur in all thesauri and
    outputs their nearest neighbours

    Prefix = directory where all thesauri are located
    Names = names of files in that directory
    If prefix is None, the names are assumed to be absolute
    """
    from thesis_generator.plugins.bov import ThesaurusVectorizer

    thesauri = []
    vect = ThesaurusVectorizer(use_pos=True, sim_threshold=0)
    for name in names:
        vect.thesaurus_file = os.path.join(prefix, name) if prefix else name
        thesauri.append(vect.load_thesaurus())
    sizes = [len(x) for x in thesauri]
    smallest_th = thesauri[sizes.index(min(sizes))]
    print 'sizes: ', sizes

    done = 0
    entries = {x for x in smallest_th.keys()}
    for th in thesauri:
        entries = entries.intersection(th.keys())

#    # compute agreement between thesauri
#    agg = numpy.zeros((len(thesauri), len(thesauri)))
#    hits = 0
#    for e in entries:
#        for id1, th1 in enumerate(thesauri):
#            for id2, th2 in enumerate(thesauri):
#                neigh1 = [x[0] for x in th1[e]]
#                neigh2 = [x[0] for x in th2[e]]
#                shared_neigh = set(neigh1).intersection(set(neigh2))
#                ranks1 = [rank for rank, x in enumerate(th1[e]) if x[0] in shared_neigh]
#                ranks2 = [rank for rank, x in enumerate(th2[e]) if x[0] in shared_neigh]
#                tau, p_value = kendalltau(ranks1, ranks2)
#                agg[id1,id2] += tau
#                hits += 1
#    print agg/float(hits)

    #get a sample of all thesauri for visual inspection
    with open('/usr/share/dict/words', 'r') as infile:
        words = [line.lower().strip() for line in infile]

    with open('thesauri-comparison.csv', 'w') as outfile:
        entries = [x for x in entries]
        while done < min(50, len(entries)):
            word = random.choice(entries)
            if word.lower().split('/')[0] not in words:
                continue
            print 'Selecting ', word
            for th, name in zip(thesauri, names):
                neigh = th[word]
                outfile.write(
                    '%s, %s,' % ('%s...%s' % (name[:8], name[-8:]), word))
                outfile.write(','.join([x[0] for x in neigh[:20]]))
                outfile.write('\n')
            outfile.write('===\n')
            done += 1


def unindex_thesauri(byblo_path, thesauri_paths):
    """
    Unindexes the entries and events files of all provided thesauri
    Parameters:
    byblo_path: Where the Byblo distribution is. Must contain tools.sh
    thesauri_paths: Iterable over Path to Byblo output directories
    """
    from iterpipes import  cmd, run
    from thesis_generator.plugins.bov import ThesaurusVectorizer

    os.chdir(byblo_path)

    for path in thesauri_paths:
        exp_name = os.path.commonprefix(glob.glob(os.path.join(path, '*')))[:-1]
        # glob with ignore the .DS_Store file on OSX
        exp_name = os.path.join(path, exp_name)
        print 'Thesaurus name is ', exp_name

        commands = []

        # events after filtering
        commands.append(cmd(
            './tools.sh unindex-events -i {}.events.filtered -o {}.events.strings -Xe {}.entry-index -Xf {}.feature-index -et JDBM',
            exp_name, exp_name, exp_name, exp_name))
        # events before filtering
        commands.append(cmd(
            './tools.sh unindex-events -i {}.events -o {}.events-unfiltered.strings -Xe {}.entry-index -Xf {}.feature-index -et JDBM',
            exp_name, exp_name, exp_name, exp_name))
        # entries after filtering
        commands.append(cmd(
            './tools.sh unindex-entries -i {}.entries.filtered -o {}.entries.strings -Xe {}.entry-index -Xf {}.feature-index -et JDBM',
            exp_name, exp_name, exp_name, exp_name))
        # entries before filtering
        commands.append(cmd(
            './tools.sh unindex-entries -i {}.entries -o {}.entries-unfiltered.strings -Xe {}.entry-index -Xf {}.feature-index -et JDBM',
            exp_name, exp_name, exp_name, exp_name))

        for cmd in commands:
            out = run(cmd)
            print "***Byblo deindex output***"
            print [x for x in out]
            print "***End Byblo output***"

#def inspect_entry_features():
#        features = {}
#        # reuse thesaurus loading code to load feature file, format is the same
#        vect = ThesaurusVectorizer(use_pos=True, sim_threshold=0, k=9999999999)
#        vect.thesaurus_file = '%s.events.strings' % exp_name
#        features[path] = vect.load_thesaurus()
#
#        # get features of 2 words which I know are neighbours sorted by frequency
#        # todo words must be chosen at random from thesaurus
#        feature_set1 = set(map(itemgetter(0), features[path]['Andres/NNP']))
#        feature_set2 = set(map(itemgetter(0), features[path]['president/NN']))
#        shared_features = feature_set1 | feature_set2
#
#        important_features1 = [x for x in features[path]['Andres/NNP'] if
#                               x[0] in shared_features]
#        important_features2 = [x for x in features[path]['president/NN'] if
#                               x[0] in shared_features]
#
#        print 'Andres/NNP: ', important_features1
#        print 'president/NN: ', important_features2
#
#        print 'hi'


def convert_old_byblo_format_to_new(filename):
    new_lines = []
    with open(filename, 'r') as infile:
        curr_token = None
        curr_line = None
        for no, line in enumerate(infile):
            line = line.strip()
            token, neigh, sim = line.split('\t')
            if not curr_token:
                curr_token = token
                curr_line = line + '\t'
            else:
                if curr_token == token:
                    curr_line += '%s\t%s\t' % (neigh, sim)
                else:
                    curr_token = token
                    new_lines.append(curr_line)
                    curr_line = line + '\t'

    new_file = '%s-new' % filename
    with open(new_file, 'w') as outfile:
        for line in new_lines:
            outfile.write(line.strip())
            outfile.write('\n')
    return new_file

if __name__ == '__main__':
#    pass
    print 'hi'
#    compare_thesauri('/Volumes/LocalScratchHD/LocalHome/Desktop/bov', [
#        'exp6-11a.strings',
#        'exp6-12a.strings',
#        'exp6-13a.strings',
#    ])

#    new_file = convert_old_byblo_format_to_new(
#        '/Volumes/LocalScratchHD/LocalHome/Desktop/thesauri_from_jack/medtest-tb-pho-no-nl-cw-55.pairs-lin.neighs-100nn')
#    compare_thesauri(None, [new_file])

    unindex_thesauri('/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/Byblo-2.1.0',
        ['/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/Byblo-2.1.0/sample-output'])
