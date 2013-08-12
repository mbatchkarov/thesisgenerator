# coding=utf-8
from unittest import TestCase
import cStringIO
from thesisgenerator.plugins.dumpers import ConsolidatedResultsCsvWriter, ConsolidatedResultsSqlWriter
import thesisgenerator.plugins.dumpers as d
from thesisgenerator.utils.misc import get_susx_mysql_conn

header_list = [x[0] for x in d.columns]
header_str = ','.join(header_list)


class TestConsolidatedResultsCsvWriter(TestCase):
    def setUp(self):
        self.fh = cStringIO.StringIO()

        # create a writer and write header
        self.writer = ConsolidatedResultsCsvWriter(self.fh)

    def test_header(self):
        self.assertEqual(header_str, self.fh.getvalue().strip())


class TestConsolidatedResultsSqlWriter(TestCase):
    def setUp(self):
        self.db_conn = get_susx_mysql_conn()

        if not self.db_conn:
            self.fail("DB connection parameters file is missing. This is "
                      "quite important")

        # create a writer and write header
        self.writer = ConsolidatedResultsSqlWriter(0, self.db_conn)

    def test_header(self):
        """
        Test if the results table for an experiment is emptied by the
        creation of a new SQL writer
        """
        q = 'SELECT * FROM data00;'
        cur = self.db_conn.cursor()
        cur.execute('SELECT * FROM data00;')
        res = cur.fetchall()
        self.assertEqual(0, len(res))

    def test_insert(self):
        """
        Test that items are correctly inserted into the database
        """
        self.writer.writerow(range(len(header_list)))

        cur = self.db_conn.cursor()
        cur.execute('SELECT * FROM data00;')
        rows = cur.fetchall()
        self.assertEqual(1, len(rows))
        self.assertEqual(len(header_list), len(rows[0]))
        for i, val in enumerate(rows[0]):
            if i == 2:
                # the third column is a timestamp, cannot convert to float
                continue
            self.assertEqual(float(i), float(val))

    def test_mysql(self):
        con = get_susx_mysql_conn()

        with con:
            cur = con.cursor()
            cur.execute("SELECT * FROM data00")
            rows = cur.fetchall()
            for row in rows:
                print row