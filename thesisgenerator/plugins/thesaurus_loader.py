# coding=utf-8
from collections import Counter
import logging
import shelve

import numpy
from thesisgenerator.plugins.tokens import DocumentFeature

from thesisgenerator.utils.misc import walk_nonoverlapping_pairs
from thesisgenerator.composers.utils import write_vectors_to_disk


class Thesaurus(object):
    def __init__(self, d):

        """
         A container that can read Byblo-formatted events (vectors) files OR sims files. Each entry can be of the form

            'water/N': [('nsubj-HEAD:title', 5), ('pobj-HEAD:by', 2)]

        i.e. entry: [(feature, count), ...], OR

            'water/N': [('horse/N', 0.5), ('earth/N', 0.4)]

        i.e. entry: [(neighbour, similarity), ...]

        :param d: a dictionary that serves as a basis
        """
        self.d = d

    def __getattr__(self, name):
        return getattr(self.d, name)

    def __setitem__(self, key, value):
        self.d[key] = value

    def __delitem__(self, key):
        del self.d[key]

    def __getitem__(self, item):
        return self.d[item]

    def __contains__(self, item):
        return item in self.d

    def __len__(self):
        return len(self.d)

    @classmethod
    def from_tsv(cls, thesaurus_files='', sim_threshold=0, include_self=False,
                 aggressive_lowercasing=True, ngram_separator='_'):
        """
        Create a Thesaurus by parsing a Byblo-compatible TSV files (events or sims).
        If duplicate values are encoutered during parsing, only the latest will be kept.

        :param thesaurus_files: list or tuple of file paths to parse
        :type thesaurus_files: list
        :param sim_threshold: min count for inclusion in this objecy
        :type sim_threshold: float
        :param include_self: whether to include self as nearest neighbour. Only applicable when holding
         similarities and not vectors
        :type include_self: bool
        :param aggressive_lowercasing: if true, most of what is read will be lowercased (excluding PoS tags), so
            Cat/N -> cat/N. This is desirable when reading full thesauri with this class. If False, no lowercasing
            will take place. This might be desirable when readings feature lists
        :type aggressive_lowercasing: bool
        :param ngram_separator: When n_gram entries are read in, what are the indidivual tokens separated by
        """
        return cls._read_from_disk(thesaurus_files,
                                   sim_threshold,
                                   include_self,
                                   ngram_separator,
                                   aggressive_lowercasing)


    @classmethod
    def _read_from_disk(cls, thesaurus_files, sim_threshold, include_self, ngram_separator, aggressive_lowercasing):
        """
        Loads a set Byblo-generated thesaurus form the specified file and
        returns their union. If any of the files has been parsed already a
        cached version is used.

        Parameters:
        thesaurus_files: string, path the the Byblo-generated thesaurus
        use_pos: boolean, whether the PoS tags should be stripped from
        entities (if they are present)
        sim_threshold: what is the min similarity for neighbours that
        should be loaded

        Returns:
        A set of thesauri or an empty dictionary
        """

        if not thesaurus_files:
            logging.warn("No thesaurus specified")
            return {}

        to_return = dict()
        for path in thesaurus_files:
            logging.info('Loading thesaurus %s from disk', path)

            FILTERED = '___FILTERED___'.lower()
            with open(path) as infile:
                for line in infile:
                    tokens = line.strip().split('\t')
                    if len(tokens) % 2 == 0:
                    # must have an odd number of things, one for the entry
                    # and pairs for (neighbour, similarity)
                        logging.warn('Dodgy line in thesaurus file: %s\n %s', path, line)
                        continue
                    if tokens[0] != FILTERED:
                        to_insert = [(_smart_lower(word, ngram_separator, aggressive_lowercasing), float(sim))
                                     for (word, sim) in walk_nonoverlapping_pairs(tokens, 1)
                                     if word.lower() != FILTERED and float(sim) > sim_threshold]
                        if include_self:
                            to_insert.insert(0, (_smart_lower(tokens[0],
                                                              ngram_separator,
                                                              aggressive_lowercasing), 1.0))
                            # the step above may filter out all neighbours of an entry. if this happens,
                            # do not bother adding it
                        if len(to_insert) > 0:
                            key = _smart_lower(tokens[0], ngram_separator, aggressive_lowercasing)
                            if DocumentFeature.from_string(key).type == 'EMPTY':
                                # do not load things in the wrong format, they'll get in the way later
                                logging.info('Skipping thesaurus entry %s', key)
                                continue

                            if key in to_return:
                                # todo this better not be a neighbours file, merging doesn't work there
                                logging.warn('Multiple entries for "%s" found. Merging.' % tokens[0])
                                c = Counter(dict(to_return[key]))
                                # print len(c)
                                c.update(dict(to_insert))
                                # print len(to_insert), len(c)
                                # print '---'
                                to_return[key] = [(k, v) for k, v in c.iteritems()]
                            else:
                                to_return[key] = to_insert

                                # note- do not attempt to lowercase if the thesaurus
                                #  has not already been lowercased- may result in
                                # multiple neighbour lists for the same entry
        return Thesaurus(to_return)

    def to_shelf(self, filename):
        logging.info('Shelving thesaurus of size %d to %s', len(self), filename)
        d = shelve.open(filename, flag='c') # read and write
        for entry, features in self.iteritems():
            d[str(entry)] = features
        d.close()

    def to_dissect_sparse_files(self, output_prefix, row_transform=None):
        """
        Converting to a dissect sparse matrix format. Writes out 3 files

        :param output_prefix: str, a
        :param row_transform:
        """
        with open('{0}.rows'.format(output_prefix), 'w+b') as outfile:
            for entry in self.keys():
                outfile.write('{}\n'.format(row_transform(entry) if row_transform else entry))

        with open('{0}.sm'.format(output_prefix), 'w+b') as outfile:
            for entry in self.keys():
                tmp_entry = row_transform(entry) if row_transform else entry
                for feature, count in self[entry]:
                    outfile.write('{} {} {}\n'.format(tmp_entry, feature, count))

        # write dissect columns file
        columns = set(feature for vector in self.values() for (feature, count) in vector)
        with open('{}.cols'.format(output_prefix), 'w+b') as outfile:
            for feature in sorted(columns):
                outfile.write('{}\n'.format(feature))

    def to_sparse_matrix(self, row_transform=None, dtype=numpy.float):
        """
        Converts the vectors held in this object to a scipy sparse matrix
        :return: a tuple containing
            1) the sparse matrix, in which rows correspond to the order of this object's iteritems()
            2) a sorted list of all features (column labels of the matrix)
            3) a sorted list of all entries (row labels of the matrix)
        :rtype: tuple
        """
        from sklearn.feature_extraction import DictVectorizer

        self.v = DictVectorizer(sparse=True, dtype=dtype)

        # order in which rows are iterated is not guaranteed if the dict if modified, but we're not doing that,
        # so it's all fine
        mat = self.v.fit_transform([dict(fv) for fv in self.itervalues()])
        rows = [k for k in self.iterkeys()]
        if row_transform:
            rows = map(row_transform, rows)

        return mat, self.v.feature_names_, rows

    def to_dissect_core_space(self):
        from composes.matrix.sparse_matrix import SparseMatrix
        from composes.semantic_space.space import Space

        mat, cols, rows = self.to_sparse_matrix()
        mat = SparseMatrix(mat)
        s = Space(mat, rows, cols)

        # test that the mapping from string to its vector has not been messed up
        for i in range(min(10, len(self))):
            s1 = s.get_row(rows[i]).mat
            s2 = self.v.transform(dict(self[rows[i]]))
            # sparse matrices do not currently support equality testing
            assert abs(s1 - s2).nnz == 0

        return s

    def to_file(self, filename, entry_filter=lambda x: True, row_transform=lambda x: x):
        """
        Writes this thesaurus to a Byblo-compatible events file like the one it was most likely read from. In the
        process converts all entries to a DocumentFeature.
        :param filename:
        :param entry_filter: Called for every DocumentFeature that is an entry in this thesaurus. The vector will
         only be written if this callable return true
        :param row_transform: Callable, any transformation that might need to be done to each entry before converting
         it to a DocumentFeature. This is needed because some entries (e.g. african/J:amod-HEAD:leader) are not
         directly convertible (needs to be african/J leader/N)
        :return: :rtype:
        """
        mat, cols, rows = self.to_sparse_matrix(row_transform=row_transform)
        rows = [DocumentFeature.from_string(x) for x in rows]
        write_vectors_to_disk(mat.tocoo(), rows, cols, filename, entry_filter=entry_filter)
        return filename


# END OF CLASS
def _smart_lower(words_with_pos, separator='_', aggressive_lowercasing=True):
    """
    Lowercase just the words and not their PoS tags
    """
    if not aggressive_lowercasing:
        return words_with_pos

    unigrams = words_with_pos.split(separator)
    words = []
    for unigram in unigrams:
        try:
            word, pos = unigram.split('/')
        except ValueError:
            # no pos
            word, pos = words_with_pos, ''

        words.append('/'.join([word.lower(), pos]) if pos else word.lower())

    return separator.join(words)
