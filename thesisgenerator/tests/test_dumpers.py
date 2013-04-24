import sqlite3
from unittest import TestCase
import cStringIO
from thesisgenerator.plugins.dumpers import ConsolidatedResultsCsvWriter, \
    ConsolidatedResultsSqliteWriter
import thesisgenerator.plugins.dumpers as d

header_list = [x[0] for x in d.columns]
header_str = ','.join(header_list)


class TestConsolidatedResultsCsvWriter(TestCase):
    def setUp(self):
        self.fh = cStringIO.StringIO()

        # create a writer and write header
        self.writer = ConsolidatedResultsCsvWriter(self.fh)

    def test_header(self):
        self.assertEqual(header_str, self.fh.getvalue().strip())


class TestConsolidatedResultsSqliteWriter(TestCase):
    def setUp(self):
        self.db_conn = sqlite3.connect(':memory:')
        # create a writer and write header
        self.writer = ConsolidatedResultsSqliteWriter(0, self.db_conn)

    def test_header(self):
        """
        Test if an emtpy table was created
        """
        q = 'SELECT name FROM sqlite_master WHERE type=\'table\';'
        res = self.db_conn.cursor().execute(q)
        tables = [x[0] for x in res]
        self.assertIn('data00', tables)

        q = 'SELECT * FROM data00;'
        res = self.db_conn.cursor().execute('SELECT * FROM data00;')
        rows = [x[0] for x in res]
        self.assertEqual(0, len(rows))

    def test_header(self):
        """
        Test that items are correctly inserted into the database
        """
        self.writer.writerow(range(len(header_list)))
        res = self.db_conn.cursor().execute('SELECT * FROM data00;')
        rows = [x for x in res]
        self.assertEqual(1, len(rows))
        self.assertEqual(len(header_list), len(rows[0]))
        for i, val in enumerate(rows[0]):
            self.assertEqual(float(i), float(val))
