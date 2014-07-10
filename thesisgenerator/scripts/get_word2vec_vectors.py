# -*- coding: utf-8 -*-
import argparse
from operator import itemgetter
import os, sys
import random
import gensim, logging, errno

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')


from discoutils.tokens import DocumentFeature
from discoutils.thesaurus_loader import Thesaurus, Vectors
from thesisgenerator.plugins.tokenizers import XmlTokenizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# inputs
conll_data_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/data/gigaword-afe-split-tagged-parsed/gigaword/'
pos_only_data_dir = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/data/gigaword-afe-split-pos/gigaword/'
# outputs
unigram_events_file = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/word2vec_vectors/thesaurus/word2vec.events.filtered.strings'
unigram_thes_file = '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/word2vec_vectors/word2vec.unigram.thesaurus.txt'

# operation parameters
MIN_COUNT = 50
WORKERS = 4
pos_map = XmlTokenizer.pos_coarsification_map


def compute_and_write_vectors(stages):
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

    if 'reformat' in stages:
        try:
            os.makedirs(pos_only_data_dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(pos_only_data_dir):
                pass
            else:
                raise

        for filename in os.listdir(conll_data_dir):
            outfile_name = os.path.join(pos_only_data_dir, filename)
            logging.info('Reformatting %s to %s', filename, outfile_name)
            with open(os.path.join(conll_data_dir, filename)) as infile, open(outfile_name, 'w') as outfile:
                for line in infile:
                    if not line.strip():  # conll empty line = sentence boundary
                        outfile.write('.\n')
                        continue
                    idx, word, lemma, pos, ner, dep, _ = line.strip().split('\t')
                    outfile.write('%s/%s ' % (lemma.lower(), pos_map[pos]))

    if 'vectors' in stages:
        # train a word2vec model
        class MySentences(object):
            def __init__(self, dirname):
                self.dirname = dirname

            def __iter__(self):
                for fname in os.listdir(self.dirname):
                    for line in open(os.path.join(self.dirname, fname)):
                        yield line.split()


        logging.info('Training word2vec')
        sentences = MySentences(pos_only_data_dir)  # a memory-friendly iterator
        model = gensim.models.Word2Vec(sentences, workers=WORKERS, min_count=MIN_COUNT)

        logging.info(model.most_similar('computer/N', topn=20))
        logging.info(model.similarity('computer/N', 'software/N'))

        entry_sample = select_entries(model.vocab.keys())


        # get word2vec vectors for each word, write to TSV
        vectors = dict()
        dimension_names = ['word2vec_feat%02d' % i for i in range(100)]  # word2vec produces 100-dim vectors
        for word in model.vocab.keys():
            # watch for non-DocumentFeatures, these break to_tsv
            # also ignore words with non-ascii characters
            if DocumentFeature.from_string(word).type == 'EMPTY' or has_non_ascii(word):
                logging.info('Ignoring vector for %s', word)
                continue
            vectors[word] = zip(dimension_names, model[word])
        th1 = Vectors(vectors)
        th1.to_tsv(unigram_events_file)

    if 'eval' in stages:
        disk_vectors = Vectors.from_tsv(unigram_events_file)
        for word in entry_sample:
            sorted_vector = sorted(disk_vectors[word], key=itemgetter(0))
            logging.info('Read from disk %s', word)
            logging.info('sorted vector %r', sorted_vector[:5])
            logging.info('matrix %r', disk_vectors.get_vector(word).A.ravel()[:5])
            logging.info('In memory')
            logging.info('word2vec value : %r', model[word][:5])
            logging.info('thes matrix value: %r', th1.get_vector(word).A.ravel()[:5])
            logging.info('thes value %r\n\n', vectors[word][:5])



    if 'thesaurus' in stages:
        # build a thesaurus out of the nearest neighbours of each unigram and save it to TSV
        # this is a little incompatible with the rest of my thesauri as it uses
        # the first PoS-augmented form for each word2vec word is used
        # nevertheless, it's useful to compare these neighbours to Byblo's neighbours as a sanity check
        logging.info('Building mini thesaurus')
        mythes = dict()
        for word in entry_sample:
            neighours = model.most_similar(word, topn=10)
            if any(has_non_ascii(foo[0]) for foo in neighours):
                continue
            mythes[word] = neighours
        Thesaurus(mythes).to_tsv(unigram_thes_file)

def select_entries(mylist):
    result = []
    for word in random.sample(mylist, 10):
        if DocumentFeature.from_string(word).type == 'EMPTY' or has_non_ascii(word):
            continue
        result.append(word)
    return result

def has_non_ascii(word):
    try:
        word.decode('ascii')
    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        return True
    return False

if __name__ == '__main__':

    assert has_non_ascii('å')
    assert not has_non_ascii('a')

    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', choices=('reformat', 'vectors', 'eval', 'thesaurus'),
                        required=True, nargs='+')
    compute_and_write_vectors(parser.parse_args().stages)

