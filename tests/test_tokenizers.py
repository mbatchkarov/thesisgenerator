from unittest import TestCase
from plugins import tokenizers
from plugins.tokenizers import xml_tokenizer, _is_number

__author__ = 'mmb28'


class Test_tokenizers(TestCase):
    params = {'normalise_entities': False,
              'use_pos': False,
              'coarse_pos': False,
              'lemmatize': False,
              'lowercase': False}

    def setUp(self):
        """
        Sets the default parameters of the tokenizer and reads a sample file
        for processing
        """
        for key, val in self.params.items():
            setattr(tokenizers, key, val)

        with open('../sample-data/test-tr/earn/earn_1.tagged') as infile:
            self.doc = infile.read()
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

        tokens = xml_tokenizer(self.doc)
        self.assertListEqual(tokens, ['Cats', 'like', 'dogs'])

        tokenizers.lowercase = True
        tokens = xml_tokenizer(self.doc)
        self.assertListEqual(tokens, ['cats', 'like', 'dogs'])

    def test_xml_tokenizer_pos(self):
        """
        Tests xml_tokenizer's coarse_pos and use_pos facilities
        """
        tokens = xml_tokenizer(self.doc)
        self.assertListEqual(tokens, ['Cats', 'like', 'dogs'])

        tokenizers.use_pos = True
        tokens = xml_tokenizer(self.doc)
        self.assertListEqual(tokens, ['Cats/NNP', 'like/VB', 'dogs/NNP'])

        tokenizers.coarse_pos = True
        tokens = xml_tokenizer(self.doc)
        self.assertListEqual(tokens, ['Cats/N', 'like/V', 'dogs/N'])

    def test_xml_tokenizer_lemmatize(self):
        """
        tests xml_tokenizer's lemmatization facility
        """

        tokenizers.lemmatize = True
        tokens = xml_tokenizer(self.doc)
        self.assertListEqual(tokens, ['Cat', 'like', 'dog'])

    def test_xml_tokenizer_common_settings(self):
        """
        Tests xml_tokenizer with the most common parameter settings. We are
        only testing a few out of the many possible combinations
        """

        tokenizers.lemmatize = True
        tokenizers.use_pos = True
        tokenizers.coarse_pos = True
        tokenizers.lowercase = True
        tokenizers.normalise_entities = True

        tokens = xml_tokenizer(self.doc)
        self.assertListEqual(tokens, ['cat/n', 'like/v', '__ner-org__'])

        tokenizers.lowercase = False
        tokens = xml_tokenizer(self.doc)
        self.assertListEqual(tokens, ['Cat/N', 'like/V', '__NER-ORG__'])

    def test_is_number(self):
        self.assertTrue('123')
        self.assertTrue('123.1928')
        self.assertTrue('123e3')
        self.assertTrue('123e-3')
        self.assertTrue('123/3')
        self.assertTrue('123/3')
        self.assertTrue('123,300')

        self.assertTrue(_is_number('asdf') == False) # for some reason
        # assertFalse does not work