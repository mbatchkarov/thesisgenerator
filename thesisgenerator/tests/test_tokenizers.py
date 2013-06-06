# coding=utf-8
from unittest import TestCase
from thesisgenerator.plugins import thesaurus_loader
from thesisgenerator.plugins.tokenizers import XmlTokenizer

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class Test_tokenizer(TestCase):
    params = {'normalise_entities': False,
              'use_pos': False,
              'coarse_pos': False,
              'lemmatize': False,
              'lowercase': False,
              'keep_only_IT': False}

    def setUp(self):
        """
        Sets the default parameters of the tokenizer and reads a sample file
        for processing
        """

        self.tokenizer = XmlTokenizer(**self.params)

        for key, val in self.params.items():
            setattr(self.tokenizer, key, val)

        with open('thesisgenerator/resources/test-tr/earn/earn_1.tagged') \
            as infile:
            self.doc = infile.read()
        with open('thesisgenerator/resources/test-ev/earn/earn_2.tagged') \
            as infile:
            self.other_doc = infile.read()

        self.assertIn('<document>', self.doc)
        self.assertIn('</document>', self.doc)
        self.assertIn('<token id=', self.doc)
        self.assertIn('</token>', self.doc)

    def test_setUp_method(self):
        """
        Tests if the setUp method has set all the parameter values to false
        """
        for key, val in self.params.items():
            self.assertFalse(val)

    def test_xml_tokenizer_lowercasing(self):
        """
        tests xml_tokenizer's lowercasing facility
        """

        tokens = self.tokenizer(self.doc)
        self.assertListEqual(tokens, ['Cats', 'like', 'dogs'])

        self.tokenizer.lowercase = True
        tokens = self.tokenizer(self.doc)
        self.assertListEqual(tokens, ['cats', 'like', 'dogs'])

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
        tokens = self.tokenizer(ET.tostring(tree))
        self.assertListEqual(tokens, ['Cats', 'dogs'])

    def test_xml_tokenizer_short_words(self):
        """
        tests xml_tokenizer's short word removal facility
        """
        self.tokenizer.lemmatize = True
        self.tokenizer.remove_short_words = True
        tokens = self.tokenizer(self.doc)
        self.assertListEqual(tokens, ['like'])

    def test_xml_tokenizer_pos(self):
        """
        Tests xml_tokenizer's coarse_pos and use_pos facilities
        """
        tokens = self.tokenizer(self.doc)
        self.assertListEqual(tokens, ['Cats', 'like', 'dogs'])

        self.tokenizer.use_pos = True
        tokens = self.tokenizer(self.doc)
        self.assertListEqual(tokens, ['Cats/NNP', 'like/VB', 'dogs/NNP'])

        self.tokenizer.coarse_pos = True
        tokens = self.tokenizer(self.doc)
        self.assertListEqual(tokens, ['Cats/N', 'like/V', 'dogs/N'])

    def test_xml_tokenizer_keep_IT_only(self):
        """
        tests xml_tokenizer's ability to discard out-of-thesaurus tokens
        """
        self.tokenizer.keep_only_IT = True
        self.tokenizer.coarse_pos = True
        self.tokenizer.use_pos = True
        self.tokenizer.lowercase = True
        self.tokenizer.lemmatize = True
        self.tokenizer.thes_entries = None

        thesaurus_loader.read_thesaurus(
            thesaurus_files=['thesisgenerator/resources/exp0-0a.strings'],
            sim_threshold=0,
            k=10,
            include_self=False)

        tokens = self.tokenizer(self.doc)
        self.assertListEqual(tokens, ['cat/n', 'like/v', 'dog/n'])

        self.tokenizer.thes_entries = None
        tokens = self.tokenizer(self.other_doc)
        self.assertListEqual(tokens, ['like/v', 'fruit/n'])


    def test_xml_tokenizer_lemmatize(self):
        """
        tests xml_tokenizer's lemmatization facility
        """

        self.tokenizer.lemmatize = True
        tokens = self.tokenizer(self.doc)
        self.assertListEqual(tokens, ['Cat', 'like', 'dog'])

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

        tokens = self.tokenizer(self.doc)
        self.assertListEqual(tokens, ['cat/n', 'like/v', '__ner-org__'])

        self.tokenizer.lowercase = False
        tokens = self.tokenizer(self.doc)
        self.assertListEqual(tokens, ['Cat/N', 'like/V', '__NER-ORG__'])

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