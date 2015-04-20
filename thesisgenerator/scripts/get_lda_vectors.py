import logging, gensim
from os.path import join
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary, TextCorpus
from thesisgenerator.scripts.get_word2vec_vectors import MySentences

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    prefix = '/mnt/lustre/scratch/inf/mmb28/FeatureExtractionToolkit'
    composed_output_dir = join(prefix, 'word2vec_vectors', 'composed')
    conll_data_dir = join(prefix, 'data/gigaword-afe-split-tagged-parsed/gigaword/')

    """background_corpus = TextCorpus(input=join(conll_data_dir, 'xaaaaaaaaaafs.tagged.conll.parsed'))
    background_corpus.dictionary.save_as_text('my_dict.dict')
    # lsi = gensim.models.lsimodel.LsiModel(corpus=background_corpus,
    # id2word=background_corpus.dictionary,
    #                                       num_topics=100)
    lda = LdaMulticore(corpus=background_corpus,
                       id2word=background_corpus.dictionary,
                       num_topics=100, workers=4)
    lda.print_topics(10)
    lda.save('ldatest.pkl')"""

    dictionary = Dictionary.load_from_text('my_dict.dict')
    lda = LdaMulticore.load('ldatest.pkl')
    print(lda[dictionary.doc2bow(['nn'])])

    # corpus = MySentences(conll_data_dir, file_percentage=20)
    # dictionary = Dictionary(corpus) # todo corpus reader currently returns sentences, not documents
    # lsi = gensim.models.lsimodel.LsiModel(corpus=corpus, num_topics=100)
    # lsi.print_topics(10)
    # lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=100, update_every=0, passes=10)
    # lda.print_topics(10)

