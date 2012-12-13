import subprocess
import os

__author__ = 'mmb28'

def stanford_process_path(path, stanfor_dir):
    """
    Puts the specified directory (in mallet format, class per subdirectory,
    depth = 1) through the stanford pipeline 'tokenize,ssplit,pos,lemma'
    Output XML that needs to be parsed.

    Paths must not contain training slashes
    """
    outputdir = '%s-tagged' % path
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    os.chdir(stanfor_dir)
    subdirs = os.listdir(path)
    for subdir in subdirs:
        outSubDir = os.path.join(outputdir, subdir)
        if not os.path.exists(outSubDir):
            os.mkdir(outSubDir)
        cmd = ['./corenlp.sh', '-annotators', 'tokenize,ssplit,pos,lemma',
               '-file', os.path.join(path,subdir), '-outputDirectory', outSubDir]

        print 'Running ', cmd
        process = subprocess.Popen(cmd)
        process.wait()

if __name__ == '__main__':
    stanford_process_path(
        '/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator/sample-data/reuters21578',
        '/Volumes/LocalScratchHD/LocalHome/Downloads/stanford-corenlp-full-2012-11-12')
