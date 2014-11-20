# -*- coding: utf-8 -*-
import argparse
import os, sys, math
from os.path import join
import gensim, logging, errno

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from discoutils.tokens import DocumentFeature
from discoutils.thesaurus_loader import Vectors
from thesisgenerator.plugins.tokenizers import pos_coarsification_map
from thesisgenerator.scripts.dump_all_composed_vectors import compose_and_write_vectors
from thesisgenerator.utils.data_utils import get_all_corpora
from thesisgenerator.composers.vectorstore import (AdditiveComposer, MultiplicativeComposer,
                                                   LeftmostWordComposer, RightmostWordComposer)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# inputs
prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit'
conll_data_dir = join(prefix, 'data/gigaword-afe-split-tagged-parsed/gigaword/')
pos_only_data_dir = join(prefix, 'data/gigaword-afe-split-pos/gigaword/')
pos_only_data_dir = join(prefix, 'data/gigaword-afe-split-pos/gigaword-small-files/')
# outputs
unigram_events_file = join(prefix, 'word2vec_vectors/word2vec-%.5fperc.unigr.strings')
composed_output_dir = join(prefix, 'word2vec_vectors', 'composed')

# unigram extraction parameters
MIN_COUNT = 50
WORKERS = 10

# composition parameters
composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]


def compute_and_write_vectors(stages, percent, repeat):
    # <markdowncell>

    # Data formatting
    # =========
    # `word2vec` produces vectors for words, such as `computer`, whereas the rest of my experiments assume there are
    # augmented with a PoS tag, e.g. `computer/N`. To get around that, start with a directory of conll-formatted
    # files such as
    #
    # ```
    # 1	Anarchism	Anarchism	NNP	MISC	5	nsubj
    # 2	is	be	VBZ	O	5	cop
    # 3	a	a	DT	O	5	det
    # 4	political	political	JJ	O	5	amod
    # 5	philosophy	philosophy	NN	O	0	root
    # ```
    #
    # and convert them to pos-augmented format (using coarse tags like Petrov's):
    #
    # ```
    # Anarchism/N is/V a/DET ....
    # ```

    # <codecell>
    global unigram_events_file
    unigram_events_file = unigram_events_file % percent

    if 'reformat' in stages:
        try:
            os.makedirs(pos_only_data_dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(pos_only_data_dir):
                pass
            else:
                raise

        for filename in os.listdir(conll_data_dir):
            outfile_name = join(pos_only_data_dir, filename)
            logging.info('Reformatting %s to %s', filename, outfile_name)
            with open(join(conll_data_dir, filename)) as infile, open(outfile_name, 'w') as outfile:
                for line in infile:
                    if not line.strip():  # conll empty line = sentence boundary
                        outfile.write('.\n')
                        continue
                    idx, word, lemma, pos, ner, dep, _ = line.strip().split('\t')
                    outfile.write('%s/%s ' % (lemma.lower(), pos_coarsification_map[pos]))

    if percent < 100 and repeat > 1:
        repeat = 1  # only repeat when using the entire corpus
    for i in range(repeat):
        if 'vectors' in stages:
            # train a word2vec model
            class MySentences(object):
                def __init__(self, dirname, file_percentage):
                    self.dirname = dirname
                    self.limit = file_percentage / 100

                def __iter__(self):
                    files = [x for x in sorted(os.listdir(self.dirname)) if not x.startswith('.')]
                    count = math.ceil(self.limit * len(files))
                    logging.info('Will use %d files for training', count)
                    for fname in files[:count]:
                        for line in open(join(self.dirname, fname)):
                            yield line.split()

            logging.info('Training word2vec on %d percent of %s', percent, pos_only_data_dir)
            sentences = MySentences(pos_only_data_dir, percent)
            model = gensim.models.Word2Vec(sentences, workers=WORKERS, min_count=MIN_COUNT)

            # get word2vec vectors for each word, write to TSV
            vectors = dict()
            dimension_names = ['f%02d' % i for i in range(100)]  # word2vec produces 100-dim vectors
            for word in model.vocab.keys():
                # watch for non-DocumentFeatures, these break to_tsv
                # also ignore words with non-ascii characters
                if DocumentFeature.from_string(word).type == 'EMPTY':
                    logging.info('Ignoring vector for %s', word)
                    continue
                vectors[word] = zip(dimension_names, model[word])
            vectors = Vectors(vectors)
            vectors.to_tsv(unigram_events_file + '.rep%d' % i, gzipped=True)

        if 'compose' in stages:
            # if we'll also be composing we don't have to write the unigram vectors to disk
            # just to read them back later.
            compose_and_write_vectors(vectors if 'vectors' in stages else unigram_events_file + '.rep%d' % i,
                                      'word2vec_%dpercent-rep%d' % (percent, i),
                                      get_all_corpora(),  # todo it is redundant to read in all corpora
                                      composer_algos,
                                      output_dir=composed_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', choices=('reformat', 'vectors', 'compose'),
                        required=True, nargs='+')
    # percent of files to use. SGE makes it easy for this to be 1, 2, ...
    parser.add_argument('--percent', default=100, type=int)
    # multiplier for args.percent. Set to 0.1 to use fractional percentages of corpus
    parser.add_argument('--multiplier', default=1., type=float)

    parser.add_argument('--repeat', default=1, type=int)
    args = parser.parse_args()
    compute_and_write_vectors(args.stages, args.percent * args.multiplier, args.repeat)

