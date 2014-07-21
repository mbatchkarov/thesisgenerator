import logging
from glob import glob
import os
import re

INPUT_DIR = '/Volumes/LocalDataHD/mmb28/Downloads/techtc100'
OUTPUT_DIR = '/Volumes/LocalDataHD/mmb28/Downloads/techtc100-clean'
DOC_REGEX = re.compile('\<dmoz_doc\>(.*?)\</dmoz_doc\>', re.DOTALL)
SUBDOC_REGEX = re.compile('\<dmoz_subdoc\>(.*?)\</dmoz_subdoc\>', re.DOTALL)


def do_single_dataset_category(dataset_name, class_name):
    dataset_outdir = os.path.join(OUTPUT_DIR, dataset_name, class_name)
    if not os.path.exists(dataset_outdir):
        os.makedirs(dataset_outdir)

    with open(os.path.join(INPUT_DIR, dataset_name, '%s.txt' % class_name)) as infile:
        data = infile.readlines()

    docs = []
    text = ''.join(data)
    for doc in DOC_REGEX.findall(text):
        subdocs = [subdoc for subdoc in SUBDOC_REGEX.findall(doc)]
        if subdocs and any(doc for doc in subdocs):
            docs.append('\n'.join(subdocs))

    for i, text in enumerate(docs):
        with open(os.path.join(dataset_outdir, '%d.txt' % i), 'w') as outfile:
            outfile.write(text)


for dataset_dir in glob('%s/*' % INPUT_DIR):
    dataset_name = os.path.basename(dataset_dir)
    print(dataset_name)

    do_single_dataset_category(dataset_name, 'all_pos')
    do_single_dataset_category(dataset_name, 'all_neg')

