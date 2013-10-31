# coding=utf-8
from unittest import TestCase
from joblib import Memory
from thesisgenerator.classifiers import NoopTransformer
from thesisgenerator.plugins.tokenizers import XmlTokenizer, Token

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class Test_tokenizer(TestCase):
    def setUp(self):
        """
        Sets the default parameters of the tokenizer and reads a sample file
        for processing
        """
        self.params = {'normalise_entities': False,
                       'use_pos': False,
                       'coarse_pos': False,
                       'lemmatize': False,
                       'lowercase': False}

        self.tokenizer = XmlTokenizer(**self.params)

        for key, val in self.params.items():
            setattr(self.tokenizer, key, val)

        with open('thesisgenerator/resources/test-tr/earn/earn_1.tagged') \
            as infile:
            self.doc = infile.read()
        with open('thesisgenerator/resources/test-ev/earn/earn_2.tagged') \
            as infile:
            self.other_doc = infile.read()

        self.doc_name = 'test_tokenizers_doc1'
        self.other_doc_name = 'test_tokenizers_doc2'

        self.assertIn('<document>', self.doc)
        self.assertIn('</document>', self.doc)
        self.assertIn('<token id=', self.doc)
        self.assertIn('</token>', self.doc)

    def test_xml_tokenizer_with_corpus_caching(self):
        from tempfile import mkdtemp

        joblib_cache_dir = mkdtemp(prefix='joblib_cache', suffix='thesgen')
        cache_memory = Memory(cachedir=joblib_cache_dir, verbose=0)
        false_memory = NoopTransformer()

        corpus = [self.doc] # a corpus of one document
        for using_joblib, memory in enumerate([false_memory, cache_memory]):
            self.params['memory'] = memory
            tokenizer = XmlTokenizer(**self.params)

            # tokenize the same corpus repeatedly
            for j in range(10):
                # get only the first sentence of the corpus
                tokenised_docs = self._strip_dependency_tree(tokenizer.tokenize_corpus(corpus, self.doc_name)[0])
                self.assertListEqual(tokenised_docs, [[Token('Cats', '', 1),
                                                       Token('like', '', 2),
                                                       Token('dogs', '', 3)]])
                # a corpus is a list of documents, which is a list of sentences, which is a list of tokens

                if using_joblib:
                    # with caching the tokenizer must only ever have one cache miss- the first time it is called
                    # if tests have been run before in this directory, the cache will still be there and no cache
                    # misses will occur
                    self.assertLessEqual(tokenizer.cache_miss_count, 1)
                else:
                    # without caching the tokenizer must miss every time
                    self.assertEqual(tokenizer.cache_miss_count, j + 1)

            # modify the tokenizer and check if cache still works as expected
            self.assertFalse(tokenizer.use_pos)
            #self.assertFalse(tokenizer.important_params['use_pos'])
            tokenizer.use_pos = True
            self.assertTrue(tokenizer.use_pos)
            #self.assertTrue(tokenizer.important_params['use_pos'])

            tokenised_docs = self._strip_dependency_tree(tokenizer.tokenize_corpus(corpus, self.doc_name)[0])
            self.assertListEqual(tokenised_docs, [[Token('Cats', 'NNP', 1),
                                                   Token('like', 'VB', 2),
                                                   Token('dogs', 'NNP', 3)]])

            # changing the parameters of the tokenizer should cause a cache miss
            self.assertEqual(tokenizer.cache_miss_count, 2 if using_joblib else j + 2)

            tokenizer.coarse_pos = True
            tokenised_docs = self._strip_dependency_tree(tokenizer.tokenize_corpus(corpus, self.doc_name)[0])
            self.assertListEqual(tokenised_docs, [[Token('Cats', 'N', 1),
                                                   Token('like', 'V', 2),
                                                   Token('dogs', 'N', 3)]])
            # another parameter change, another cache miss
            self.assertEqual(tokenizer.cache_miss_count, 3 if using_joblib else j + 3)

            # changing a parameter back to what is was should cause a cache hit
            tokenizer.use_pos = False
            tokenizer.coarse_pos = False
            tokenised_docs = self._strip_dependency_tree(tokenizer.tokenize_corpus(corpus, self.doc_name)[0])
            self.assertListEqual(tokenised_docs, [[Token('Cats', '', 1),
                                                   Token('like', '', 2),
                                                   Token('dogs', '', 3)]])
            self.assertEqual(tokenizer.cache_miss_count, 3 if using_joblib else j + 4)

        import shutil

        shutil.rmtree(joblib_cache_dir)

    def test_xml_tokenizer_lowercasing(self):
        """
        tests xml_tokenizer's lowercasing facility
        """

        tokens = self._strip_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('Cats', '', 1),
                                       Token('like', '', 2),
                                       Token('dogs', '', 3)]])

        self.params['lowercase'] = True
        self.tokenizer = XmlTokenizer(**self.params)
        tokens = self._strip_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('cats', '', 1),
                                       Token('like', '', 2),
                                       Token('dogs', '', 3)]])

    def test_xml_tokenizer_stopwords(self):
        """
                tests xml_tokenizer's stopword removal facility
        """

        # replace one of the words with a stopword
        tree = ET.fromstring(self.doc.encode("utf8"))
        for element in tree.findall('.//token'):
            txt = element.find('lemma').text
            if txt == 'like':
                element.find('lemma').text = 'the'
                element.find('word').text = 'the'

        self.tokenizer.remove_stopwords = True
        tokens = self._strip_dependency_tree(self.tokenizer.tokenize_doc(ET.tostring(tree)))
        self.assertListEqual(tokens, [[Token('Cats', '', 1),
                                       Token('dogs', '', 3)]])

        self.tokenizer.use_pos = True
        self.tokenizer.coarse_pos = True
        tokens = self._strip_dependency_tree(self.tokenizer.tokenize_doc(ET.tostring(tree)))
        self.assertListEqual(tokens, [[Token('Cats', 'N', 1),
                                       Token('dogs', 'N', 3)]])

    def test_xml_tokenizer_short_words(self):
        """
        tests xml_tokenizer's short word removal facility
        """
        self.tokenizer.lemmatize = True
        self.tokenizer.remove_short_words = True
        tokens = self._strip_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('like', '', 2)]])

        self.tokenizer.use_pos = True
        self.tokenizer.coarse_pos = True
        tokens = self._strip_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('like', 'V', 2)]])

    def test_xml_tokenizer_pos(self):
        """
        Tests xml_tokenizer's coarse_pos and use_pos facilities
        """
        tokens = self._strip_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('Cats', '', 1),
                                       Token('like', '', 2),
                                       Token('dogs', '', 3)]])

        self.tokenizer.use_pos = True
        tokens = self._strip_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('Cats', 'NNP', 1),
                                       Token('like', 'VB', 2),
                                       Token('dogs', 'NNP', 3)]])

        self.tokenizer.coarse_pos = True
        tokens = self._strip_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('Cats', 'N', 1),
                                       Token('like', 'V', 2),
                                       Token('dogs', 'N', 3)]])

    def test_xml_tokenizer_lemmatize(self):
        """
        tests xml_tokenizer's lemmatization facility
        """

        self.tokenizer.lemmatize = True
        tokens = self._strip_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('Cat', '', 1),
                                       Token('like', '', 2),
                                       Token('dog', '', 3)]])

    def test_xml_tokenizer_common_settings(self):
        """
        Tests xml_tokenizer with the most common parameter settings. We are
        only testing a few out of the many possible combinations
        """

        self.tokenizer.lemmatize = True
        self.tokenizer.use_pos = True
        self.tokenizer.coarse_pos = True
        self.tokenizer.lowercase = True
        self.tokenizer.normalise_entities = True

        tokens = self._strip_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[
                                          Token('cat', 'N', 1),
                                          Token('like', 'V', 2),
                                          Token('__NER-ORG__', '', 3)
                                      ]])

        self.tokenizer.lowercase = False
        tokens = self._strip_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[
                                          Token('Cat', 'N', 1),
                                          Token('like', 'V', 2),
                                          Token('__NER-ORG__', '', 3)
                                      ]])

    def _strip_dependency_tree(self, tokens):
        return [sent[0] for sent in tokens]

    def test_is_number(self):
        is_number = self.tokenizer._is_number

        self.assertTrue(is_number('123'))
        self.assertTrue(is_number('123.1928'))
        self.assertTrue(is_number('123e3'))
        self.assertTrue(is_number('123e-3'))
        self.assertTrue(is_number('123/3'))
        self.assertTrue(is_number('123/3'))
        self.assertTrue(is_number('123,300'))

        self.assertFalse(is_number('asdf'))
        # todo tests for dependency-parsed input data