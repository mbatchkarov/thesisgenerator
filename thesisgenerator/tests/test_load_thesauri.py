# coding=utf-8
from unittest import TestCase
from thesisgenerator.plugins import thesaurus_loader
from thesisgenerator.plugins.thesaurus_loader import _smart_lower
from thesisgenerator.utils.misc import walk_nonoverlapping_pairs

__author__ = 'mmb28'


class TestLoad_thesauri(TestCase):
    def setUp(self):
        """
        Sets the default parameters of the tokenizer and reads a sample file
        for processing
        """

        self.params = {
            'thesaurus_files': ['thesisgenerator/resources/exp0-0a.strings'],
            'sim_threshold': 0,
            # 'k': 10,
            'include_self': False
        }
        self.thesaurus = thesaurus_loader.Thesaurus(**self.params)

    def _reload_thesaurus(self):
        self.thesaurus = thesaurus_loader.Thesaurus(**self.params)

    def test_empty_thesaurus(self):
        self.params['thesaurus_files'] = []
        self._reload_thesaurus()
        self._reload_and_assert(0, 0)

        # should raise KeyError for unknown tokens
        with self.assertRaises(KeyError):
            self.thesaurus['kasdjhfka']

    def _reload_and_assert(self, entry_count, neighbour_count):
        th = thesaurus_loader.Thesaurus(**self.params)
        all_neigh = [x for v in th.values() for x in v]
        self.assertEqual(len(th), entry_count)
        self.assertEqual(len(all_neigh), neighbour_count)
        return th

    def test_sim_threshold(self):
        for i, j, k in zip([0, .39, .5, 1], [7, 3, 3, 0], [14, 4, 3, 0]):
            self.params['sim_threshold'] = i
            self._reload_thesaurus()
            self._reload_and_assert(j, k)

    # def test_k(self):
    #     for i, j, k in zip([3, 2, 1, 0], [7, 7, 7, 0], [14, 12, 7, 0]):
    #         self.params['k'] = i
    #         self._reload_thesaurus()
    #         self._reload_and_assert(j, k)

    def test_include_self(self):
        for i, j, k in zip([False, True], [7, 7], [14, 21]):
            self.params['include_self'] = i
            self._reload_thesaurus()
            th = self._reload_and_assert(j, k)

            for entry, neighbours in th.items():
                self.assertIsInstance(entry, str)
                self.assertIsInstance(neighbours, list)
                self.assertIsInstance(neighbours[0], tuple)
                if i:
                    self.assertEqual(entry, neighbours[0][0])
                    self.assertEqual(1, neighbours[0][1])
                else:
                    self.assertNotEqual(entry, neighbours[0][0])
                    self.assertGreaterEqual(1, neighbours[0][1])

    def test_smart_lower(self):
        # test that the PoS of an n-gram entry is not lowercased
        self.assertEquals(_smart_lower('Cat/N'), 'cat/N')
        self.assertEquals(_smart_lower('Cat/n'), 'cat/n')
        self.assertEquals(_smart_lower('Red/J CaT/N'), 'red/J cat/N')

        # test that features are not touched
        self.assertEquals(_smart_lower('amod-DEP:former', aggressive_lowercasing=False), 'amod-DEP:former')

    def test_iterate_nonoverlapping_pairs(self):
        inp = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        output1 = [x for x in walk_nonoverlapping_pairs(inp, 1)]
        self.assertListEqual([(1, 2), (3, 4), (5, 6), (7, 8)], output1)