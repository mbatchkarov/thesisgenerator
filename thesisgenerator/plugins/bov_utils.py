import glob
import os
import shutil
import traceback
from concurrent.futures import as_completed
from numpy import nonzero
import sys


try:
    from thesisgenerator.__main__ import go
    from thesisgenerator.utils import replace_in_file
except ImportError:
# if one tries to run this script from the main project directory the
# thesisgenerator package would not be on the path, add it and try again
    sys.path.append('../../')
    sys.path.append('../')
    sys.path.append('./')
    sys.path.append('./thesisgenerator')
    from thesisgenerator.__main__ import go
    from thesisgenerator.utils import replace_in_file

__author__ = 'mmb28'

def inspect_thesaurus_effect(outdir, clf_name, thesaurus_file, pipeline,
                             predicted, x_test):
    """
    Evaluates the performance of a classifier with and without the thesaurus
    that backs its vectorizer
    """

    # remove the thesaurus
    pipeline.named_steps['vect'].thesaurus = {}
    predicted2 = pipeline.predict(x_test)

    with open('%s/before_after_%s.csv' % (outdir, thesaurus_file),
              'a') as outfile:
        outfile.write('DocID,')
        outfile.write(','.join([str(x) for x in range(len(predicted))]))
        outfile.write('\n')
        outfile.write('%s+Thesaurus,' % clf_name)
        outfile.write(','.join([str(x) for x in predicted.tolist()]))
        outfile.write('\n')
        outfile.write('%s-Thesaurus,' % clf_name)
        outfile.write(','.join([str(x) for x in predicted2.tolist()]))
        outfile.write('\n')
        outfile.write('Decisions changed: %d' % (
            nonzero(predicted - predicted2)[0].shape[0]))
        outfile.write('\n')


def _do_single_thesaurus(conf_file, id, t, test_data, train_data):
    name, ext = os.path.splitext(conf_file)
    name = '%s-variants' % name
    if not os.path.exists(name):
        os.mkdir(name)
    new_conf_file = os.path.join(name, 'run%d%s' % (id, ext))
    log_file = os.path.join(name, '%d.log' % (id))
    shutil.copy(conf_file, new_conf_file)
    configspec_file = os.path.join(os.path.dirname(conf_file), '.confrc')
    shutil.copy(configspec_file, name)

    replace_in_file(new_conf_file, 'name=.*', 'name=run%d' % id)
    replace_in_file(new_conf_file, 'training_data=.*',
                    'training_data=%s' % train_data)
    replace_in_file(new_conf_file, 'test_data=.*',
                    'test_data=%s' % test_data)
    # it is important that the list of thesaurus files in the conf file ends with a comma
    replace_in_file(new_conf_file, 'thesaurus_files=.*',
                    'thesaurus_files=%s,' % t)
    return go(new_conf_file, log_file)


def evaluate_thesauri(pattern, conf_file, train_data='sample-data/web-tagged',
                      test_data='sample-data/web2-tagged', pool_size=1):
    from concurrent.futures import ProcessPoolExecutor

    thesauri = glob.glob(pattern)

    #Create a number (processes) of individual processes for executing parsers.
    with ProcessPoolExecutor(max_workers=pool_size) as executor:
        jobs = {} #Keep record of jobs and their input
        id = 1
        for t in thesauri:
            future = executor.submit(_do_single_thesaurus, conf_file, id, t,
                                     test_data, train_data)
            jobs[future] = id
            id += 1

        #As each job completes, check for success, print details of input
        for job in as_completed(jobs.keys()):
            try:
                status, outfile = job.result()
                print("Success. Files produced: %s" % outfile)
            except Exception as exc:
                print(
                    " Exception encountered in: \n-- %s" % "\n-- ".join(
                        jobs[job]))
                print(''.join(traceback.format_exception(*sys.exc_info())))
                raise exc

if __name__ == '__main__':
    evaluate_thesauri(
        '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/Byblo-2.1.0/exp6-*d/*90sims.neighbours.strings',
        '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/conf/main.conf',
        pool_size=4)
