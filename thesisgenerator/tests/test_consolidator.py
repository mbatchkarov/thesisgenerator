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

        exp = {
            'total_tok': 9, 'total_typ': 5,
            'iv_it_tok_mean': 3, 'iv_it_tok_std': 0,
            'iv_oot_tok_mean': 0, 'iv_oot_tok_std': 0,
            'oov_it_tok_mean': 2, 'oov_it_tok_std': 0,
            'oov_oot_tok_mean': 4, 'oov_oot_tok_std': 0,
            'iv_it_typ_mean': 2, 'iv_it_typ_std': 0,
            'iv_oot_typ_mean': 0, 'iv_oot_typ_std': 0,
            'oov_it_typ_mean': 1, 'oov_it_typ_std': 0,
            'oov_oot_typ_mean': 2, 'oov_oot_typ_std': 0
        }
        for variable, exp_value in exp.items():
            cursor.execute('SELECT DISTINCT %s from data00;' % variable)
            res = cursor.fetchall()
            self.assertEqual(res[0][0], exp_value)




