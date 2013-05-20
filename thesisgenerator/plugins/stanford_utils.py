# coding=utf-8
import os
import sys
import subprocess
import traceback
import datetime as dt
import xml.etree.cElementTree as ET

from concurrent.futures import ProcessPoolExecutor, as_completed


################
#
# Utilities
#
################
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
    jobs = {}
    with ProcessPoolExecutor(max_workers=processes) as executor:
        for data_file in os.listdir(path_to_data):
            if not (data_file.startswith(".") or data_file.endswith(".conll")):
                input_path = os.path.join(path_to_data, data_file)
                jobs[executor.submit(_process_single_xml_to_conll,
                                     input_path)] = data_file
        for job in as_completed(jobs):
            try:
                job.result() #Propagates any exceptions.
            except Exception as e:
                logger.print_info(
                    " Exception during formatting: %s" % jobs[job])
                logger.print_info(
                    ''.join(traceback.format_exception(*sys.exc_info())),
                    printstdin=False)
                raise
            logger.print_info(" Formatting complete: %s" % jobs[job])
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
                for word, lemma, pos in zip(element.findall(".//word"),
                                            element.findall(".//lemma"),
                                            element.findall(".//POS")):
                    outfile.write("%s\t%s\t%s\t%s\n" % (
                        i, word.text.encode('utf8'), lemma.text.encode('utf8'),
                        pos.text))
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
    sys.path.append(liblinear_path)

    def chunks(items, no_of_chunks):
        """Split *items* into a number (no_of_chunks) of equal chunks."""
        chunksize = (len(items) + no_of_chunks // 2) // no_of_chunks
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
        with ProcessPoolExecutor(max_workers=processes) as executor:
            jobs = {} #Keep record of jobs and their input
            #Split data into chunks, submit a parsing job for each chunk
            for files in chunks([name for name in os.listdir(input_sub_dir) if
                                 not name.startswith('.') and name.endswith(
                                         ".conll")], processes):
                jobs[executor.submit(run_parser, input_sub_dir, files,
                                     output_sub_dir,
                                     parser_project_path)] = files
                #As each job completes, check for success, print details of input
            for job in as_completed(jobs.keys()):
                try:
                    pfiles, info = job.result()
                    logger.print_info(
                        " Success. Files processed: \n-- %s" % "\n-- ".join(
                            pfiles))
                    logger.print_info(info)
                except Exception as exc:
                    logger.print_info(
                        " Exception encountered in: \n-- %s" % "\n-- ".join(
                            jobs[job]))
                    logger.print_info(
                        ''.join(traceback.format_exception(*sys.exc_info())),
                        printstdin=False)
                    raise

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
def execute_pipeline(path_to_corpora, #Required for all
                     path_to_stanford="", #Required for stanford pipeline
                     path_to_filelistdir="", #optional
                     path_to_depparser="", #Required for dependency parsing
                     log="", #optional
                     path_to_liblinear="", #Required if liblinear not in path
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
    """
    Email Andy or Miro for a copy of the readme for this script
    """

    #Pipeline examples:
    #    run = set("stanford formatting parsing cleanup".split())
    run = set("stanford".split())

    #Fill arguments below, for example:
    execute_pipeline(
        '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/sample-data/sts',
        path_to_stanford='/Volumes/LocalScratchHD/LocalHome/Downloads/stanford-corenlp-full-2012-11-12',
        stanford_java_threads=8,
        run=run)
