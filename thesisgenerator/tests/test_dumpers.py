# coding=utf-8
from unittest import TestCase
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
from thesisgenerator.plugins.dumpers import ConsolidatedResultsCsvWriter
import thesisgenerator.plugins.dumpers as d



class TestConsolidatedResultsCsvWriter(TestCase):
    def setUp(self):
        self.fh = StringIO()

        # create a writer and write header
        self.writer = ConsolidatedResultsCsvWriter(self.fh)

    def test_header(self):
        header_list = [x[0] for x in d.columns]
        header_str = ','.join(header_list)
        self.assertEqual(header_str, self.fh.getvalue().strip())


