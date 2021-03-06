import sys

sys.path.append('.')
import logging
from os.path import join
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary, TextCorpus
from gensim.models import LsiModel, TfidfModel
from sklearn.preprocessing import normalize
from discoutils.misc import mkdirs_if_not_exists
from thesisgenerator.scripts.get_word2vec_vectors import (MySentences, get_args_from_cmd_line,
                                                          write_gensim_vectors_to_tsv)
from thesisgenerator.composers.vectorstore import (compose_and_write_vectors, RightmostWordComposer,
                                                   LeftmostWordComposer, MultiplicativeComposer,
                                                   AdditiveComposer, VerbComposer)


class WordPosCorpusReader(TextCorpus):
    def __init__(self, dirname, file_percentage, repeat_num=0):
        self.metadata = False
        self.sentences = MySentences(dirname, file_percentage, repeat_num=repeat_num)
        self.dictionary = Dictionary(documents=self.get_texts())
        # todo more LDA params here
        self.dictionary.filter_extremes(keep_n=20000, no_above=.25, no_below=20)  # todo keep size of vocab in check
        # we're using doc=sentence definition, so we can be generous with the minimum document frequency
        self.dictionary.compactify()

    def get_texts(self):
        yield from self.sentences


class LdaToWord2vecAdapter(object):
    def __init__(self, lda, dictionary):
        self.dictionary = dictionary

        self.topics = lda.state.get_lambda()
        # normalise by row. shape is (100, 30k). Use L1 norm to make row sum to 1
        self.topics = normalize(self.topics, axis=0, norm='l1')

    def __getitem__(self, word):
        return self.topics[:, self.dictionary.token2id[word]]


def train_lda_model(data_dir, unigram_events_file, tmp_file_prefix, percent):
    corpus = WordPosCorpusReader(data_dir, percent)
    tfidf = TfidfModel(corpus)
    dictionary = corpus.dictionary
    dictionary.save_as_text('%s_dict.txt' % tmp_file_prefix)
    logging.info('LDA dictionary is %s', dictionary)
    lda = LdaMulticore(corpus=tfidf[corpus],
                       id2word=dictionary,  # this MUST be there, can't be set automatically from corpus. WTF?
                       num_topics=100, workers=10, passes=5)  # todo control params of LDA here
    lda.save('%s_lda.model' % tmp_file_prefix)
    vectors = write_gensim_vectors_to_tsv(LdaToWord2vecAdapter(lda, dictionary),
                                          unigram_events_file,
                                          vocab=dictionary.token2id.keys())
    return lda, dictionary, vectors


def main(corpus_name, stages, percent):
    prefix = '/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'
    composed_output_dir = join(prefix, 'lda_vectors', 'composed')
    mkdirs_if_not_exists(composed_output_dir)
    composer_algos = [AdditiveComposer, MultiplicativeComposer, LeftmostWordComposer,
                      RightmostWordComposer, VerbComposer]

    if corpus_name == 'gigaw':
        data_dir = join(prefix, 'data/gigaword-afe-split-pos/gigaword/')
        unigram_events_file = join(prefix, 'lda_vectors/lda-gigaw-%dperc.unigr.strings' % percent)
        tmp_file_prefix = 'lda_gigaw_%dper' % percent
    elif corpus_name == 'wiki':
        data_dir = join(prefix, 'data/wikipedia-tagged-pos/wikipedia/')
        unigram_events_file = join(prefix, 'lda_vectors/lda-wiki-%dperc.unigr.strings' % percent)
        tmp_file_prefix = 'lda_wiki_%dper' % percent


    # lsi = LsiModel(corpus=corpus, id2word=dictionary, num_topics=100)
    # lsi.save('lsitest.pkl')

    # lsi = LsiModel.load('lsitest.pkl')
    # dictionary = Dictionary.load_from_text('lsidict.txt')
    # logging.info(dictionary)
    # lsi.print_topics(10)
    # doc_bow = dictionary.doc2bow(['return'])
    # logging.info('LSI vector is %r', lsi[doc_bow])
    # logging.info('-------------------')

    if 'vectors' in stages:
        lda, dictionary, vectors = train_lda_model(data_dir, unigram_events_file, tmp_file_prefix, percent)
    else:
        logging.info('Not training LDA model, loading pretrained instead')
        lda = LdaMulticore.load('%s_lda.model' % tmp_file_prefix)
        dictionary = Dictionary.load_from_text('%s_dict.txt' % tmp_file_prefix)
        vectors = unigram_events_file
    lda.print_topics(10)

    if 'compose' in stages:
        if vectors is None:
            vectors = LdaToWord2vecAdapter(lda, dictionary)
        out_path = 'lda-%s_%dpercent' % (corpus_name, percent)
        compose_and_write_vectors(vectors,
                                  out_path,
                                  composer_algos,
                                  output_dir=composed_output_dir,
                                  row_filter=lambda x, y: True,
                                  dense_hd5=True)
        # logging.info('Log perplexity is %f', lda.log_perplexity(corpus))
        # doc_bow = dictionary.doc2bow('return/V recur/V baker/N bustling/J'.split())
        # logging.info('LDA vectors is %r', lda[doc_bow])

        # for word in sample(dictionary.token2id.keys(), 50):
        # doc_bow = dictionary.doc2bow([word])
        # logging.info('LDA vector for %s is %r', word, lda[tfidf[doc_bow]])

        # topics = lda.state.get_lambda()
        # # normalise by row. shape is (100, 30k)
        # topics = normalize(topics, axis=0, norm='l1')
        # for word in sample(dictionary.token2id.keys(), 10):
        # vector = topics[:, dictionary.token2id[word]]
        # logging.info('LDA vector for %s is %r. Total mass %1.1f', word, vector, vector.sum())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t%(levelname)s : %(message)s")
    args = get_args_from_cmd_line()
    main(args.corpus, args.stages, args.percent)




