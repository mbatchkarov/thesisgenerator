# coding=utf-8
from math import ceil
import os
import sys
import subprocess
import datetime as dt
import xml.etree.cElementTree as ET


################
#
# Utilities
#
################
from joblib import Parallel, delayed


def current_time(): #for reporting purposes.
    return dt.datetime.ctime(dt.datetime.now())


def _make_filelist_and_create_files(data_dir, filelistpath, output_dir):
    """
    1. Create a list of files in a directory to be processed, which
       can be passed to stanford's "filelist" input argument.
    2. Pre-create each output file in an attempt to avoid cluster
       problems.
    """
    with open(filelistpath, 'w') as filelist:
        for filename in os.listdir(data_dir):
            if not filename.startswith("."):
                filepath = os.path.join(data_dir, filename)
                filelist.write("%s\n" % filepath)
                with open(os.path.join(output_dir, filename + ".tagged"),
                          'w'):
                    pass


class Logger(object):
    """Use for program-wide printing to logfile."""

    def __init__(self):
        self.log = ""

    def flush(self):
        sys.stdout.flush()
        if self.log:
            self.log.flush()

    def set_log(self, path):
        if path:
            self.log = open(path, 'a')

    def print_info(self, info, flush=True, printstdin=True):
        if self.log:
            self.log.write(info + "\n")
        if printstdin:
            print info
        if flush:
            self.flush()

##########
#
# Create a logger
#
##########
logger = Logger()


#####################
#
# Process raw text through stanford pipeline
#
#####################
def run_stanford_pipeline(data_dir, stanford_dir, java_threads=2,
                          filelistdir=""):
    """
    Process directory of text using stanford core nlp
    suite. Perform:
        - Tokenisation
        - Sentence segmentation
        - PoS tagging
        - Lemmatisation

    Output XML to "*data_dir*-tagged"
    """
    if not all([data_dir, stanford_dir]):
        raise ValueError("ERROR: Must specify path to data and stanford tools.")

    #Create output directory
    output_dir = "%s-tagged" % data_dir
    try:
        os.mkdir(output_dir)
    except OSError:
        pass #Directory already exists

    #Change working directory to stanford tools
    os.chdir(stanford_dir)

    logger.print_info("<%s> Beginning stanford pipeline..." % current_time())

    for data_sub_dir in [name for name in os.listdir(data_dir) if
                         not name.startswith(".")]:
        #Setup output subdirectory
        output_sub_dir = os.path.join(output_dir, data_sub_dir)
        input_sub_dir = os.path.join(data_dir, data_sub_dir)
        try:
            os.mkdir(output_sub_dir)
        except OSError:
            pass #Directory already exists

        #Create list of files to be processed.
        filelist = os.path.join(filelistdir if filelistdir else stanford_dir,
                                "%s-filelist.txt" % data_sub_dir)
        _make_filelist_and_create_files(input_sub_dir, filelist, output_sub_dir)

        logger.print_info("<%s> Beginning stanford processing: %s" % (
            current_time(), input_sub_dir))

        #Construct stanford java command.
        stanford_cmd = ['./corenlp.sh', '-annotators',
                        'tokenize,ssplit,pos,lemma,ner',
                        # '-file', input_sub_dir, '-outputDirectory', output_sub_dir,
                        '-filelist', filelist, '-outputDirectory',
                        output_sub_dir,
                        '-threads', str(java_threads), '-outputFormat', 'xml',
                        '-outputExtension', '.tagged']

        logger.print_info("Running: \n" + str(stanford_cmd))

        #Run stanford script, block until complete.
        subprocess.call(stanford_cmd)

        logger.print_info("<%s> Stanford complete for path: %s" % (
            current_time(), output_sub_dir))

    logger.print_info("<%s> All stanford complete." % current_time())

    return output_dir

##################
#
#  Formatting to CoNLL from XML format
#
##################
def process_corpora_from_xml(path_to_corpora, processes=1):
    """
    Given a directory of corpora, where each corpus is a
    directory of xml files produced by stanford_pipeline,
    convert text to CoNLL-style formatting:
        ID    FORM    LEMMA    POS
    Jobs are run in parallel.
    """
    logger.print_info("<%s> Starting XML conversion..." % current_time())
    for data_sub_dir in os.listdir(path_to_corpora):
        _process_xml_to_conll(os.path.join(path_to_corpora, data_sub_dir),
                              processes)


def _process_xml_to_conll(path_to_data, processes=1):
    """
    Given a directory of XML documents from stanford's output,
    convert them to CoNLL style sentences. Jobs run in parallel.
    """
    logger.print_info("<%s> Beginning formatting to CoNLL: %s" % (
        current_time(), path_to_data))
    # jobs = {}
    Parallel(n_jobs=processes)(delayed(_process_single_xml_to_conll)(
        os.path.join(path_to_data, data_file))
                               for data_file in os.listdir(path_to_data)
                               if not (data_file.startswith(".") or
                                       data_file.endswith(".conll")))

    logger.print_info("<%s> All formatting complete." % current_time())


def _process_single_xml_to_conll(path_to_file):
    """
    Convert a single file from XML to CoNLL style.
    """
    with open(path_to_file + ".conll", 'w') as outfile:
        #Create iterator over XML elements, don't store whole tree
        xmltree = ET.iterparse(path_to_file, events=("end",))
        for _, element in xmltree:
            if element.tag == "sentence": #If we've read an entire sentence
                i = 1
                #Output CoNLL style
                for word, lemma, pos, ner in zip(element.findall(".//word"),
                                                 element.findall(".//lemma"),
                                                 element.findall(".//POS"),
                                                 element.findall(".//NER")):
                    outfile.write("%s\t%s\t%s\t%s\t%s\n" % (
                        i, word.text.encode('utf8'), lemma.text.encode('utf8'),
                        pos.text, ner.text))
                    i += 1
                outfile.write("\n")
                #Clear this section of the XML tree
                element.clear()

####################
#
#  Dependency Parsing
#
####################
def dependency_parse_directory(data_dir, parser_project_path, liblinear_path,
                               processes=20):
    """Dependency parse conll style data, in several simultaneous processes."""

    #Add to python path location of dependency parser and liblinear
    sys.path.append(os.path.join(parser_project_path, "src"))

    def chunks(items, no_of_chunks):
        """Split *items* into a number (no_of_chunks) of equal chunks."""
        chunksize = int(ceil((len(items) + no_of_chunks / 2.) / no_of_chunks))
        return (items[i:i + chunksize] for i in xrange(0, len(items),
                                                       chunksize))

    #Create output directory
    output_dir = "%s-parsed" % data_dir
    try:
        os.mkdir(output_dir)
    except OSError:
        pass #Directory already exists

    logger.print_info("<%s> Beginning dependency parsing..." % current_time())

    for data_sub_dir in [path for path in os.listdir(data_dir) if
                         not path.startswith('.')]:
        output_sub_dir = os.path.join(output_dir, data_sub_dir)
        input_sub_dir = os.path.join(data_dir, data_sub_dir)
        #Create output corpus directory
        try:
            os.mkdir(output_sub_dir)
        except OSError:
            pass #directory already exists

        logger.print_info("<%s> Parsing: %s" % (current_time(), input_sub_dir))

        #Create a number (processes) of individual processes for executing parsers.
        files_chunks = chunks([name for name in os.listdir(input_sub_dir) if
                               not name.startswith('.') and
                               name.endswith(".conll")], processes)
        files_chunks = list(files_chunks)

        # for f in files_chunks:
        #     run_parser(input_sub_dir, f, output_sub_dir,parser_project_path)

        # run parsing in parallel
        Parallel(n_jobs=processes)(delayed(run_parser)(input_sub_dir,
                                                       f,
                                                       output_sub_dir,
                                                       parser_project_path)
                                   for f in files_chunks)

        logger.print_info("<%s> Parsing Complete." % current_time())


def run_parser(input_dir, input_files, output_dir, parser_project_path):
    """Create a parser and parse a list of files"""
    from parsing.parsing_functions import DependencyParser

    start = dt.datetime.now()
    input_filepaths = [os.path.join(input_dir, name) for name in input_files]
    output_filepaths = [os.path.join(output_dir, name + ".parsed") for name in
                        input_files]
    dp = DependencyParser() #Shit just got real
    dp.parse_file_list(input_filepaths,
                       output_filepaths,
                       os.path.join(parser_project_path, "examples",
                                    "model_files", "penn-stanford-index"),
                       os.path.join(parser_project_path, "examples",
                                    "model_files", "penn-stanford-model"),
                       format_file=os.path.join(parser_project_path, "examples",
                                                "feature_extraction_toolkit_type.txt")
    )
    info = " %s file(s) processed, time taken: %s" % (
        len(input_files), dt.datetime.now() - start)
    return input_files, info

#############
#
# Cleaning up
#
############
def remove_temp_files(path_to_corpora):
    """Remove XML versions of processed data"""
    logger.print_info("<%s> Removing XML files..." % current_time())
    for data_sub_dir in os.listdir(path_to_corpora):
        dir_path = os.path.join(path_to_corpora, data_sub_dir)
        if not data_sub_dir.startswith(".") and os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if not filename.startswith(".") and not filename.endswith(
                        "conll"):
                    os.remove(os.path.join(dir_path, filename))
    logger.print_info("<%s> XML files removed." % current_time())

###################
#
# Methods of invocation
#
##################
def execute_pipeline(path_to_corpora, # Required for all
                     path_to_stanford="", # Required for stanford pipeline
                     path_to_filelistdir="", # optional
                     path_to_depparser="", # Required for dependency parsing
                     log="", # optional
                     path_to_liblinear="", # Required if liblinear not in path
                     run=frozenset(["stanford", "formatting", "parsing"]),
                     stanford_java_threads=40,
                     formatting_python_processes=40,
                     parsing_python_processes=40
):
    logger.set_log(log)

    if "stanford" in run:
        if not path_to_stanford:
            raise ValueError("Specify path to stanford")
        run_stanford_pipeline(path_to_corpora, path_to_stanford,
                              stanford_java_threads, path_to_filelistdir)

    tagged_path = path_to_corpora + "-tagged"

    if "formatting" in run:
        process_corpora_from_xml(tagged_path, formatting_python_processes)

    if "parsing" in run:
        if not path_to_depparser:
            raise ValueError(
                "Specify path to dependency parser")
        dependency_parse_directory(tagged_path, path_to_depparser,
                                   path_to_liblinear, parsing_python_processes)

    if "cleanup" in run:
        remove_temp_files(tagged_path)

    logger.print_info("Pipeline finished")


if __name__ == "__main__":
    '''
    ---- Resources Required ----

    This section lists software required to
    run annotate_corpora.py.

    	1. AR's Dependency Parsing Project
    	2. Python 2.6 or Python 2.7
    	3. Liblinear installation with Python interface
    	4. Stanford CoreNLP pipeline
    	5. Python's joblib package

    	- How to acquire resources:
    		1. AR's Dependency Parsing Project
    			- Clone git repository at .../data3/adr27/DEPPARSE/parse_suite_repo, or
    			- Ask for a copy, or
    			- Github coming soon.

    		2. Python 2.6 of Python 2.7
    			- Download from: http://www.python.org/download/releases/<VERSION>/

    		3. Liblinear installation with Python interface
    			- Download liblinear package from:
    				http://www.csie.ntu.edu.tw/~cjlin/liblinear/
    			- CD to extracted directory. Make.
    			- CD to python directory within. Make.
    			- Create a "liblinear" folder in your Python installation's site-packages
    			- Copy liblinear.py and liblinearutil.py from the python folder to site-packages
    			- Copy liblinear.so.1 from the main liblinear download directory to site-packages
    			- Create empty "__init__.py" file in site-packages
    			- Open liblinear.py, around line 19, there's be a path like:
    				'../liblinear.so.1'
    			  Change it to "./liblinear.so.1" (the new location of it in your site-packages)
    			- IMPORTANT NOTE: Ensure you perform the make process on the machine you intend
    			  to run liblinear on. And place in the relevant python site-packages.

    		4. Stanford CoreNLP pipeline
    			- Download from: http://nlp.stanford.edu/software/corenlp.shtml

    		5. Joblib
    			- Download from: http://pypi.python.org/pypi/joblib or
    			install with pip


    ---- Execution ----

    This section explains how to run stanford_utils.py

    	- Expected Input

    		The pipeline expects input data in the following structure:
    			- A directory containing corpora, where
    			- Each corpus is a directory of files, where
    			- Each file contains raw text.

    	- Output

    		After running the full pipeline on a directory called "corpora"
    		You should see the following output:

    			- A directory called "corpora-tagged" contains a version of
    			  your data in CoNLL style format after the execution of the
    			  following parts of stanford corenlp:

    			  	- Tokenization
    			  	- Sentence segmenation
    			  	- Lemmatisation
    			  	- PoS tagging

    			- A directory called "corpora-tagged-parsed" which adds the
    			  annotations of AR's dependency parser to the data.

    	- Invokation using "execute_pipeline" function

    		This function allows you to run all or individual parts of the pipeline.

    		Currently, you should call the function with the appropriate parameters
    		by writing the arguments in a call to the function at the bottom of the
    		script.

    		Perhaps a config script parser would make things better...

    		It requires the following arguments (some are optional with defaults):

    		- run
    			A sequence or collection of strings which specify which parts of the
    			pipeline to run. There are 4 options:

    			stanford   : run the stanford pipeline
    			formatting : convert stanford XML output to CoNLL
    			parsing    : dependency parse CoNLL format text
    			cleanup    : delete stanford XML files

    		- path_to_corpora
    			This is the full path to the directory containing your corpora.

    		- path_to_stanford
    			This is the full path to the directory containing Stanford CoreNLP

    		- path_to_filelistdir
    			Before running Stanford, a list of files to be processed is created
    			and saved to disk, to be passed as an argument to stanford. This
    			is the path to the DIRECTORY where this file should be saved (1 per
    			corpus)

    			DEFAULT: The stanford corenlp directory

    		- stanford_java_threads
    			The number of threads to be used when running stanford corenlp

    			DEFAULT: 40

    		- formatting_python_processes
    			The number of python processes to run in parallel when
    			converting XML to CoNLL.

    			DEFAULT: 40

    		- parsing_python_processes
    			The number of python processes to run in parallel when
    			dependency parsing. Each requires about 1-2gb of RAM.

    			DEFAULT: 40

    		- path_to_depparser
    			Path to AR's dependency parsing project.

    		- path_to_liblinear
    			Path to liblinear installation, required if not located
    			in the Python Path already.

    		- log
    			Path to logfile.

    			DEFAULT: no logging.
    '''

    #Pipeline examples:
    run = set("stanford formatting parsing cleanup".split())
    #run = set("parsing cleanup".split())

    # run = set("formatting parsing cleanup".split())
    # run = set("formatting".split())
    # run = set("parsing".split())
    # run = set("stanford".split())

    #Fill arguments below, for example:
    execute_pipeline(
        '/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/data/gigaword-afe-split',
        path_to_stanford='/mnt/lustre/scratch/inf/mmb28/FeatureExtrationToolkit/stanford-corenlp-full-2013-06-20',
        path_to_depparser='/mnt/lustre/scratch/inf/mmb28/parser_repo_miro',
        # path_to_liblinear='/Volumes/LocalDataHD/mmb28/NetBeansProjects/liblinear',
        stanford_java_threads=40,
        parsing_python_processes=40,
        run=run)

