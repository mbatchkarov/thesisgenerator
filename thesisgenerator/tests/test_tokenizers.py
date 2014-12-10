# coding=utf-8
from operator import attrgetter
from unittest import TestCase
from thesisgenerator.plugins.tokenizers import XmlTokenizer
from discoutils.tokens import DocumentFeature, Token

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

        with open('thesisgenerator/resources/tokenizer/invalid_tokens.tagged') \
                as infile:
            self.invalid_doc = infile.read()

        self.doc_name = 'test_tokenizers_doc1'
        self.other_doc_name = 'test_tokenizers_doc2'

        self.assertIn('<document>', self.doc)
        self.assertIn('</document>', self.doc)
        self.assertIn('<token id=', self.doc)
        self.assertIn('</token>', self.doc)

    def test_xml_tokenizer_lowercasing(self):
        """
        tests xml_tokenizer's lowercasing facility
        """

        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('Cats', '', 1),
                                       Token('like', '', 2),
                                       Token('dogs', '', 3)]])

        self.params['lowercase'] = True
        self.tokenizer = XmlTokenizer(**self.params)
        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('cats', '', 1),
                                       Token('like', '', 2),
                                       Token('dogs', '', 3)]])

    def test_xml_tokenizer_invalid_tokens(self):
        """
        tests xml_tokenizer's ability to remove tokens that do contain / or _ . These chars
        are used as separators later and having them in the token messes up parsing
        """

        # test that invalid tokens are removed
        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(self.invalid_doc))
        self.assertListEqual(tokens[0], [Token('like', '', 2),
                                         Token('dogs', '', 3, ner='ORG')])

        # test that valid named entity tokens get normalised
        self.params['normalise_entities'] = True
        self.tokenizer = XmlTokenizer(**self.params)
        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(self.invalid_doc))
        self.assertListEqual(tokens[0], [Token('like', '', 2),
                                         Token('__NER-ORG__', '', 3, ner='ORG')])

    def test_dependencies(self):
        self.params['use_pos'] = True
        self.params['coarse_pos'] = True
        self.params['lowercase'] = True
        self.tokenizer = XmlTokenizer(**self.params)
        self.tokenizer.remove_stopwords = True

        with open('thesisgenerator/resources/earn_1-with-deps.tagged') as infile:
            doc = infile.read()

        dep_tree = self.tokenizer.tokenize_doc(doc)[0]

        # token index is 1-indexed, as is the output of Stanford
        cats = Token('cats', 'N', 1)
        like = Token('like', 'V', 2)
        dogs = Token('dogs', 'N', 3, ner='ORG')

        self.assertSetEqual(set(dep_tree.nodes()), set([cats, like, dogs]))
        self.assertEqual(len(dep_tree.edges()), 2)
        self.assertDictEqual(dep_tree.succ, {cats: {},
                                             like: {cats: {'type': 'nsubj'}, dogs: {'type': 'dobj'}},
                                             dogs: {}})

        # remove the subject of the sentence and check that dependency is gone
        tree = self._replace_word_in_document('cat', 'the', document=doc)
        dep_tree = self.tokenizer.tokenize_doc(ET.tostring(tree))[0]

        self.assertSetEqual(set(dep_tree.nodes()), set([like, dogs]))
        self.assertEqual(len(dep_tree.edges()), 1)
        # the subject isn't there anymore
        self.assertDictEqual(dep_tree.succ, {like: {dogs: {'type': 'dobj'}},
                                             dogs: {}})


    def _replace_word_in_document(self, original, replacement, document=None):
        if not document:
            document = self.doc

        tree = ET.fromstring(document.encode("utf8"))
        for element in tree.findall('.//token'):
            txt = element.find('lemma').text
            if txt.lower() == original:
                element.find('lemma').text = replacement
                element.find('word').text = replacement
        return tree

    def test_xml_tokenizer_stopwords(self):
        """
                tests xml_tokenizer's stopword removal facility
        """

        # replace one of the words with a stopword
        tree = self._replace_word_in_document('like', 'the')

        self.tokenizer.remove_stopwords = True
        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(ET.tostring(tree)))
        self.assertListEqual(tokens, [[Token('Cats', '', 1),
                                       Token('dogs', '', 3)]])

        self.tokenizer.use_pos = True
        self.tokenizer.coarse_pos = True
        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(ET.tostring(tree)))
        self.assertListEqual(tokens, [[Token('Cats', 'N', 1),
                                       Token('dogs', 'N', 3)]])

    def test_xml_tokenizer_short_words(self):
        """
        tests xml_tokenizer's short word removal facility
        """
        self.tokenizer.lemmatize = True
        self.tokenizer.remove_short_words = True
        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('like', '', 2)]])

        self.tokenizer.use_pos = True
        self.tokenizer.coarse_pos = True
        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('like', 'V', 2)]])

    def test_xml_tokenizer_pos(self):
        """
        Tests xml_tokenizer's coarse_pos and use_pos facilities
        """
        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('Cats', '', 1),
                                       Token('like', '', 2),
                                       Token('dogs', '', 3)]])

        self.tokenizer.use_pos = True
        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('Cats', 'NNP', 1),
                                       Token('like', 'VB', 2),
                                       Token('dogs', 'NNP', 3)]])

        self.tokenizer.coarse_pos = True
        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[Token('Cats', 'N', 1),
                                       Token('like', 'V', 2),
                                       Token('dogs', 'N', 3)]])

    def test_xml_tokenizer_lemmatize(self):
        """
        tests xml_tokenizer's lemmatization facility
        """

        self.tokenizer.lemmatize = True
        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
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
        self.tokenizer.remove_stopwords = False
        self.tokenizer.remove_short_words = False

        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[
                                          Token('cat', 'N', 1),
                                          Token('like', 'V', 2),
                                          Token('dog', 'N', 3)
                                      ]])

        self.tokenizer.lowercase = False
        tokens = self._tokens_from_dependency_tree(self.tokenizer.tokenize_doc(self.doc))
        self.assertListEqual(tokens, [[
                                          Token('Cat', 'N', 1),
                                          Token('like', 'V', 2),
                                          Token('dog', 'N', 3)
                                      ]])

    def _tokens_from_dependency_tree(self, trees):
        return [sorted(tree.nodes(), key=attrgetter('index')) for tree in trees]

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