from itertools import groupby, chain
import logging
import numpy as np

__author__ = 'mmb28'


def write_vectors_to_disk(matrix, row_index, column_index, features_path, entries_path, vectors_path,
                          entry_filter=lambda x: True):
    """

    :param matrix: data matrix of size (n_entries, n_features) in scipy.sparse.coo format
    :param row_index: sorted list of DocumentFeature-s representing entry names
    :param column_index: sorted list of feature names
    :param features_path: where to write the Byblo features file
    :param entries_path: where to write the Byblo entries file
    :param vectors_path: where to write the Byblo events file
    :param entry_filter: callable, called for each entry. Returns true if the entry has to be written and false if
    the entry has to be ignored. Defaults to True.
    """
    # todo write unit test
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
            if row_num % 100 == 0:
                logging.info('Processed %d vectors', row_num)

    logging.info('Writing to %s', entries_path)
    with open(entries_path, 'w') as outfile:
        for entry, count in new_byblo_entries.iteritems():
            outfile.write('%s\t%f\n' % (entry.tokens_as_str(), count))

    logging.info('Writing to %s', features_path)
    with open(features_path, 'w') as outfile:
        if selected_rows: # guard against empty files
            feature_sums = np.array(matrix.tocsr()[selected_rows].sum(axis=0))[0, :]
            for feature, count in zip(column_index, feature_sums):
                if count > 0:
                    outfile.write('%s\t%f\n' % (feature, count))
