import sys
sys.path.append('.')
import logging
from os.path import join
from random import sample
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary, TextCorpus
from gensim.models import LsiModel, TfidfModel
from thesisgenerator.scripts.get_word2vec_vectors import MySentences


class WordPosCorpusReader(TextCorpus):
    def __init__(self, dirname, file_percentage, repeat_num=0):
        self.metadata = False
        self.sentences = MySentences(dirname, file_percentage, repeat_num=repeat_num)
        self.dictionary = Dictionary(documents=self.get_texts())
        self.dictionary.filter_extremes(keep_n=2000, no_above=.25) # can keep size of vocab in check
        self.dictionary.compactify()

    def get_texts(self):
        yield from self.sentences


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'
    composed_output_dir = join(prefix, 'word2vec_vectors', 'composed')
    pos_only_data_dir = join(prefix, 'data/gigaword-afe-split/gigaword/')

    corpus = WordPosCorpusReader(pos_only_data_dir, 10) # todo use all data
    tfidf = TfidfModel(corpus)
    dictionary = corpus.dictionary
    dictionary.save_as_text('lsidict.txt')
    logging.info(dictionary)
    # lsi = LsiModel(corpus=corpus, id2word=dictionary, num_topics=100)
    # lsi.save('lsitest.pkl')

    # lsi = LsiModel.load('lsitest.pkl')
    # dictionary = Dictionary.load_from_text('lsidict.txt')
    # logging.info(dictionary)
    # lsi.print_topics(10)
    # doc_bow = dictionary.doc2bow(['return/V'])
    # logging.info('LSI vector is %r', lsi[doc_bow])
    # logging.info('-------------------')

    lda = LdaMulticore(corpus=tfidf[corpus],
                       id2word=dictionary, # this MUST be there, can't be set automatically from corpus. WTF?
                       num_topics=100, workers=4, passes=5)
    lda.save('ldatest.pkl')

    lda = LdaMulticore.load('ldatest.pkl')
    lda.print_topics(10)
    logging.info('Log perplexity is %f', lda.log_perplexity(corpus))
    # doc_bow = dictionary.doc2bow('return/V recur/V baker/N bustling/J'.split())
    # logging.info('LDA vectors is %r', lda[doc_bow])
    for word in sample(dictionary.token2id.keys(), 50):
        doc_bow = dictionary.doc2bow([word])
        logging.info('LDA vector for %s is %r', word, lda[tfidf[doc_bow]])


