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
    thesauri = []
    vect = ThesaurusVectorizer(use_pos=True, sim_threshold=0)
    for name in names:
        vect.thesaurus_file = os.path.join(prefix, name)
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

    print words
    with open('thesauri-comparison.csv', 'w') as outfile:
        entries = [x for x in entries]
        while done < 50:
            word = random.choice(entries)
            if word.lower().split('/')[0] not in words:
                continue
            print word
            for th, name in zip(thesauri, names):
                neigh = th[word]
                outfile.write('%s, %s, ' % (name[:8], word))
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
        exp_name = os.path.commonprefix(glob.glob(os.path.join(path,'*')))[:-1]
        # glob with ignore the .DS_Store file on OSX
        exp_name = os.path.join(path, exp_name)
        print 'Thesaurus name is ',exp_name
        unindex_cmd = cmd(
            './tools.sh unindex-events -i {}.events -o '
            '{}.events.strings -Xe {}.entry-index -Xf {}.feature-index -et '
            'JDBM',
            exp_name,exp_name, exp_name, exp_name)
        out = run(unindex_cmd)
        print "***Byblo deindex output***"
        print [x for x in out]
        print "***End Byblo output***"

        features = {}
        # reuse thesaurus loading code to load feature file, format is the same
        vect = ThesaurusVectorizer(use_pos=True, sim_threshold=0,
                                   lowercase=False,k=9999999999)
        vect.thesaurus_file = '%s.events.strings'%exp_name
        features[path] = vect.load_thesaurus()

        # get features of 2 words which I know are neighbours sorted by frequency
        # todo words must be chosen at random from thesaurus
        feature_set1 = set(map(itemgetter(0), features[path]['Andres/NNP']))
        feature_set2 = set(map(itemgetter(0), features[path]['president/NN']))
        shared_features = feature_set1 & feature_set2

        important_features1 = [x for x in features[path]['Andres/NNP'] if x[0] in shared_features]
        important_features2 = [x for x in features[path]['president/NN'] if x[0] in shared_features]

        print 'Andres/NNP: ', important_features1
        print 'president/NN: ', important_features2

        print 'hi'





if __name__ == '__main__':
#    stanford_process_path(
#        '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/sample-data/web2',
#        '/Volumes/LocalScratchHD/LocalHome/Downloads/stanford-corenlp-full-2012-11-12')

#    compare_thesauri('/Volumes/LocalScratchHD/LocalHome/Desktop/bov', [
#        #        'exp6-11a.sims.neighbours.strings',
#        #        'exp6-12a.sims.neighbours.strings',
#        'exp6-13a.sims.neighbours.strings',
#        'exp6-1a.sims.neighbours.strings'
#    ])

    unindex_thesauri('/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/Byblo-2.1.0',
        ['/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/Byblo-2.1.0/output'])