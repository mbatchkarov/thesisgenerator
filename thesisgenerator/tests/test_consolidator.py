from unittest import TestCase
from thesisgenerator.plugins.experimental_utils import run_experiment

__author__ = 'mmb28'


class TestConsolidator(TestCase):
    @classmethod
    def setUpClass(cls):
        i = 0
        conf_pattern = '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/thesisgenerator/resources'
        #  %(project_dir, i, i)
        run_experiment(0, num_workers=1, prefix=conf_pattern)


    def test__extract_thesausus_coverage_info(self):
        self.fail()