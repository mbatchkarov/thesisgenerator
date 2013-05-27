from unittest import TestCase
from thesisgenerator.plugins.experimental_utils import run_experiment
from thesisgenerator.utils import get_susx_mysql_conn

__author__ = 'mmb28'


class TestConsolidator(TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = 'thesisgenerator/resources'
        run_experiment(0, num_workers=1, predefined_sized=[3],
                       prefix=prefix)

    def test_extract_thesausus_coverage_info(self):
        with open('thesisgenerator/resources/conf/exp0/output/summary0.csv') \
            as infile:
            log_txt = ''.join(infile.readlines())
        cursor = get_susx_mysql_conn().cursor()

        cursor.execute('SELECT DISTINCT classifier from data00;')
        res = cursor.fetchall()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], 'MultinomialNB')

        cursor.execute('SELECT DISTINCT '
                       'total_tok,total_typ,'
                       'iv_it_tok_mean,iv_it_tok_std,'
                       'iv_oot_tok_mean,iv_oot_tok_std,'
                       'oov_it_tok_mean,oov_it_tok_std,'
                       'oov_oot_tok_mean,oov_oot_tok_std,'
                       'iv_it_typ_mean,iv_it_typ_std,'
                       'iv_oot_typ_mean,iv_oot_typ_std,'
                       'oov_it_typ_mean,oov_it_typ_std,'
                       'oov_oot_typ_mean,oov_oot_typ_std '
                       'from data00;')
        res = cursor.fetchall()
        self.assertTupleEqual(res[0],
                              (
                                  20, 12, 12, 0, 0, 0, 2, 0, 4, 0, 6, 0, 0, 0,
                                  1, 0,
                                  3, 0))
        print res




