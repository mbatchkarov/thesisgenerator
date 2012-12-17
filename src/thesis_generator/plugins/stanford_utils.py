import glob
from operator import itemgetter
import random
import subprocess
import os
from thesis_generator.plugins.bov import ThesaurusVectorizer

__author__ = 'mmb28'

def stanford_process_path(path, stanfor_dir):
    """
    Puts the specified directory (in mallet format, class per subdirectory,
    depth = 1) through the stanford pipeline 'tokenize,ssplit,pos,lemma'
    Output XML that needs to be parsed.

    Paths must not contain training slashes
    """
    outputdir = '%s-tagged' % path
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    os.chdir(stanfor_dir)
    subdirs = os.listdir(path)
    for subdir in subdirs:
        outSubDir = os.path.join(outputdir, subdir)
        if not os.path.exists(outSubDir):
            os.mkdir(outSubDir)
        cmd = ['./corenlp.sh', '-annotators', 'tokenize,ssplit,pos,lemma,ner',
               '-file', os.path.join(path, subdir), '-outputDirectory',
               outSubDir]

        print 'Running ', cmd
        process = subprocess.Popen(cmd)
        process.wait()


def compare_thesauri(prefix, names):
    """
    Prefix = directory where all thesauri are located
    Names = names of files in that directory
    If prefix is None, the names are assumed to be absolute
    """
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

    with open('/usr/share/dict/words', 'r') as infile:
        words = [line.lower().strip() for line in infile]

    with open('thesauri-comparison.csv', 'w') as outfile:
        entries = [x for x in entries]
        while done < min(50, len(entries)):
            word = random.choice(entries)
            if word.lower().split('/')[0] not in words:
                continue
            print 'Selecting ',word
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
    Iterable over Path to Byblo output directories
    """
    from iterpipes import  cmd, run

    os.chdir(byblo_path)

    for path in thesauri_paths:
        exp_name = os.path.commonprefix(glob.glob(os.path.join(path, '*')))[:-1]
        # glob with ignore the .DS_Store file on OSX
        exp_name = os.path.join(path, exp_name)
        print 'Thesaurus name is ', exp_name
        unindex_cmd = cmd(
            './tools.sh unindex-events -i {}.events -o '
            '{}.events.strings -Xe {}.entry-index -Xf {}.feature-index -et JDBM',
            exp_name, exp_name, exp_name, exp_name)
        out = run(unindex_cmd)
        print "***Byblo deindex output***"
        print [x for x in out]
        print "***End Byblo output***"

        features = {}
        # reuse thesaurus loading code to load feature file, format is the same
        vect = ThesaurusVectorizer(use_pos=True, sim_threshold=0, k=9999999999)
        vect.thesaurus_file = '%s.events.strings' % exp_name
        features[path] = vect.load_thesaurus()

        # get features of 2 words which I know are neighbours sorted by frequency
        # todo words must be chosen at random from thesaurus
        feature_set1 = set(map(itemgetter(0), features[path]['Andres/NNP']))
        feature_set2 = set(map(itemgetter(0), features[path]['president/NN']))
        shared_features = feature_set1 | feature_set2

        important_features1 = [x for x in features[path]['Andres/NNP'] if
                               x[0] in shared_features]
        important_features2 = [x for x in features[path]['president/NN'] if
                               x[0] in shared_features]

        print 'Andres/NNP: ', important_features1
        print 'president/NN: ', important_features2

        print 'hi'


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
#    stanford_process_path(
#        '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/sample-data/web2',
#        '/Volumes/LocalScratchHD/LocalHome/Downloads/stanford-corenlp-full-2012-11-12')

    compare_thesauri('/Volumes/LocalScratchHD/LocalHome/Desktop/bov', [
        'exp6-11a.sims.neighbours.strings',
        'exp6-12a.sims.neighbours.strings',
        'exp6-13a.sims.neighbours.strings',
        'exp6-1a.sims.neighbours.strings'
    ])

#    new_file = convert_old_byblo_format_to_new(
#        '/Volumes/LocalScratchHD/LocalHome/Desktop/thesauri_from_jack/medtest-tb-pho-no-nl-cw-55.pairs-lin.neighs-100nn')
#    compare_thesauri(None, [new_file])

#    unindex_thesauri('/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/Byblo-2.1.0',
#        ['/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/Byblo-2.1.0/output'])
