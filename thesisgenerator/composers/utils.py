from itertools import groupby, chain
import logging
import numpy as np
from scipy.sparse import isspmatrix_coo

__author__ = 'mmb28'


def write_vectors_to_disk(matrix, row_index, column_index, vectors_path, features_path='', entries_path='',
                          entry_filter=lambda x: True):
    """
    Converts a matrix and its associated row/column indices to a Byblo compatible entries/features/event files,
    possibly applying a tranformation function to each entry
    :param matrix: data matrix of size (n_entries, n_features) in scipy.sparse.coo format
    :type matrix: scipy.sparse.coo_matrix
    :param row_index: sorted list of DocumentFeature-s representing entry names
    :type row_index: thesisgenerator.plugins.tokenizer.DocumentFeature
    :param column_index: sorted list of feature names
    :param features_path: where to write the Byblo features file
    :param entries_path: where to write the Byblo entries file
    :param vectors_path: where to write the Byblo events file
    :param entry_filter: callable, called for each entry. Takes a single DocumentFeature parameter. Returns true
    if the entry has to be written and false if the entry has to be ignored. Defaults to True.
    """
    # todo unit test
    if not isspmatrix_coo(matrix):
        logging.error('Expected a scipy.sparse.coo matrix, got %s', type(matrix))
        raise ValueError('Wrong matrix type')
    if (len(row_index), len(column_index)) != matrix.shape:
        logging.error('Matrix shape is wrong, expected %dx%s, got %r', len(row_index), len(column_index), matrix.shape)
        raise ValueError('Matrix shape does not match row_index/column_index size')

    new_byblo_entries = {}
    things = zip(matrix.row, matrix.col, matrix.data)
    selected_rows = []

    logging.info('Writing to %s', vectors_path)
    with open(vectors_path, 'wb') as outfile:
        for row_num, group in groupby(things, lambda x: x[0]):
            entry = row_index[row_num]
            if entry_filter(entry):
                selected_rows.append(row_num)
                ngrams_and_counts = [(column_index[x[1]], x[2]) for x in group]
                outfile.write('%s\t%s\n' % (
                    entry.tokens_as_str(),
                    '\t'.join(map(str, chain.from_iterable(ngrams_and_counts)))
                ))
                new_byblo_entries[entry] = sum(x[1] for x in ngrams_and_counts)
            if row_num % 1000 == 0:
                logging.info('Processed %d vectors', row_num)

    if entries_path:
        logging.info('Writing to %s', entries_path)
        with open(entries_path, 'w') as outfile:
            for entry, count in new_byblo_entries.iteritems():
                outfile.write('%s\t%f\n' % (entry.tokens_as_str(), count))

    if features_path:
        logging.info('Writing to %s', features_path)
        with open(features_path, 'w') as outfile:
            if selected_rows: # guard against empty files
                feature_sums = np.array(matrix.tocsr()[selected_rows].sum(axis=0))[0, :]
                for feature, count in zip(column_index, feature_sums):
                    if count > 0:
                        outfile.write('%s\t%f\n' % (feature, count))


def reformat_entries(filename, suffix, function, separator='\t'):
    # todo unit test
    """
    Applies a function to the first column of a file **in place**.
    :param filename: File to apply transformation to.
    :param function: Function to apply, takes and returns a single string.
    :param separator: The columns in the file are separated by this.
    """

    #shutil.copy(filename, filename + '.bak')
    outname = '{}{}'.format(filename, suffix)
    with open(filename) as infile, open(outname, 'w') as outfile:
        for line in infile:
            fields = line.split(separator)
            fields[0] = function(fields[0])
            outfile.write(separator.join(fields))
    return outname


def julie_transform(input, pos1='J', pos2='N', separator='_'):
    # todo unit test
    '''african/J:amod-HEAD:ancestry -> african/J_ancestry/N'''
    head = input.split(':')[-1]
    modifier = input.split('/')[0]
    return '{modifier}/{pos1}{separator}{head}/{pos2}'.format(**locals())


def julie_transform2(input, pos1='J', pos2='N', separator='_'):
    '''account/N:nn-DEP:bank -> bank/N_account/N'''
    modifier = input.split(':')[-1]
    head = input.split('/')[0]
    return '{modifier}/{pos1}{separator}{head}/{pos2}'.format(**locals())