# -*- coding: utf-8 -*-
import argparse
import os, sys, math
from os.path import join
from functools import reduce
import gensim, logging, errno
import numpy as np

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from discoutils.tokens import DocumentFeature
from discoutils.thesaurus_loader import Vectors
from thesisgenerator.plugins.tokenizers import pos_coarsification_map
from thesisgenerator.composers.vectorstore import (AdditiveComposer, MultiplicativeComposer,
                                                   LeftmostWordComposer, RightmostWordComposer,
                                                   compose_and_write_vectors)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# unigram extraction parameters
MIN_COUNT = 50
WORKERS = 10

# composition parameters
composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer, RightmostWordComposer]


def _train_model(percent, data_dir):
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

    logging.info('Training word2vec on %d percent of %s', percent, data_dir)
    sentences = MySentences(data_dir, percent)
    model = gensim.models.Word2Vec(sentences, workers=WORKERS, min_count=MIN_COUNT)
    return model


def _vectors_to_tsv(model, vocab, output_path):
    # get word2vec vectors for each word, write to TSV
    vectors = dict()
    dimension_names = ['f%02d' % i for i in range(100)]  # word2vec produces 100-dim vectors
    for word in vocab:
        # watch for non-DocumentFeatures, these break to_tsv
        # also ignore words with non-ascii characters
        if DocumentFeature.from_string(word).type == 'EMPTY':
            logging.info('Ignoring vector for %s', word)
            continue
        vectors[word] = zip(dimension_names, model[word])
    vectors = Vectors(vectors)
    vectors.to_tsv(output_path, gzipped=True)
    del model
    return vectors


def reformat_data(conll_data_dir, pos_only_data_dir):
    """
    Data formatting
    =========
    `word2vec` produces vectors for words, such as `computer`, whereas the rest of my experiments assume there are
    augmented with a PoS tag, e.g. `computer/N`. To get around that, start with a directory of conll-formatted
    files such as

    ```
    1	Anarchism	Anarchism	NNP	MISC	5	nsubj
    2	is	be	VBZ	O	5	cop
    3	a	a	DT	O	5	det
    4	political	political	JJ	O	5	amod
    5	philosophy	philosophy	NN	O	0	root
    ```

    and convert them to pos-augmented format (using coarse tags like Petrov's):

    ```
    Anarchism/N is/V a/DET ....
    ```
    :param conll_data_dir: input directory in CONLL format
    :param pos_only_data_dir: output directory
    """
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


def compute_and_write_vectors(corpus_name, stages, percent, repeat):
    prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'
    composed_output_dir = join(prefix, 'word2vec_vectors', 'composed')

    if corpus_name == 'gigaw':
        # inputs
        conll_data_dir = join(prefix, 'data/gigaword-afe-split-tagged-parsed/gigaword/')
        # pos_only_data_dir = join(prefix, 'data/gigaword-afe-split-pos/gigaword/')
        pos_only_data_dir = join(prefix, 'data/gigaword-afe-split-pos/gigaword-small-files/')
        # outputs
        unigram_events_file = join(prefix, 'word2vec_vectors/word2vec-gigaw-%dperc.unigr.strings')
    elif corpus_name == 'wiki':
        conll_data_dir = None  # wiki data is already in the right format, no point in reformatting
        pos_only_data_dir = join(prefix, 'data/wikipedia-tagged-pos/wikipedia/')
        unigram_events_file = join(prefix, 'word2vec_vectors/word2vec-wiki-%dperc.unigr.strings')
    else:
        raise ValueError('Unknown corpus %s' % corpus_name)

    unigram_events_file = unigram_events_file % percent  # fill in percentage information

    if percent < 100 and repeat > 1:
        repeat = 1  # only repeat when using the entire corpus

    if 'reformat' in stages:
        reformat_data(conll_data_dir, pos_only_data_dir)

    if 'vectors' in stages:
        models = []
        for i in range(repeat):
            model = _train_model(percent, pos_only_data_dir)
            models.append(model)
            vocab = model.vocab.keys()

        vectors = []
        # write the output of each run separately
        for i in range(repeat):
            output_path = unigram_events_file + '.rep%d' % i
            vectors.append(_vectors_to_tsv(model, vocab, output_path))

        if 'average' in stages and repeat > 1:
            # average vectors and append to list to be written
            output_path = unigram_events_file + '.avg%d' % repeat
            model = {}
            for k in vocab:
                model[k] = reduce(np.add, [m[k] for m in models])
            vectors.append(_vectors_to_tsv(model, vocab, output_path))

    if 'compose' in stages:
        for i, v in enumerate(vectors):
            # if we'll also be composing we don't have to write the unigram vectors to disk
            # just to read them back later.
            if 'average' in stages and i == (len(vectors) - 1) and len(vectors) > 1:
                # last set of vectors in the list, these are the averages ones
                out_path = 'word2vec-%s_%dpercent-avg%d' % (corpus_name, percent, repeat)
                input_thing = v if 'vectors' in stages else unigram_events_file + '.avg%d' % repeat
            else:
                out_path = 'word2vec-%s_%dpercent-rep%d' % (corpus_name, percent, i)
                input_thing = v if 'vectors' in stages else unigram_events_file + '.rep%d' % i
            compose_and_write_vectors(input_thing,
                                      out_path,
                                      composer_algos,
                                      output_dir=composed_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', choices=('reformat', 'vectors', 'average', 'compose'),
                        required=True, nargs='+')
    parser.add_argument('--corpus', choices=('gigaw', 'wiki'), required=True)
    # percent of files to use. SGE makes it easy for this to be 1, 2, ...
    parser.add_argument('--percent', default=100, type=int)
    # multiplier for args.percent. Set to 0.1 to use fractional percentages of corpus
    parser.add_argument('--repeat', default=1, type=int)
    args = parser.parse_args()
    logging.info('Params are: %r', args)
    compute_and_write_vectors(args.corpus, args.stages, args.percent, args.repeat)

