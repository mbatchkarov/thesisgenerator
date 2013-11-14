# coding=utf-8
import logging
from thesisgenerator.utils.misc import walk_nonoverlapping_pairs


class Thesaurus(dict):
    def __init__(self, thesaurus_files='', sim_threshold=0, include_self=False, aggressive_lowercasing=True):
        """
         A container that can read Byblo-formatted events (vectors) files OR sims files. Each entry can be of the form

            'water/N': [('nsubj-HEAD:title', 5), ('pobj-HEAD:by', 2)]

        i.e. entry: [(feature, count), ...], OR

            'water/N': [('horse/N', 0.5), ('earth/N', 0.4)]

        i.e. entry: [(neighbour, similarity), ...]

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
        """
        self.thesaurus_files = thesaurus_files
        self.sim_threshold = sim_threshold
        self.include_self = include_self
        self.aggressive_lowercasing = aggressive_lowercasing

        self._read_from_disk()


    def _read_from_disk(self):
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

        if not self.thesaurus_files:
            logging.warn("No thesaurus specified")
            return {}

        for path in self.thesaurus_files:
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
                        to_insert = [(_smart_lower(word, self.aggressive_lowercasing), float(sim))
                                     for (word, sim) in walk_nonoverlapping_pairs(tokens, 1)
                                     if word.lower() != FILTERED and float(sim) > self.sim_threshold]
                        if self.include_self:
                            to_insert.insert(0, (_smart_lower(tokens[0]), 1.0))
                            # the step above may filter out all neighbours of an entry. if this happens,
                            # do not bother adding it
                        if len(to_insert) > 0:
                            if tokens[0] in self:
                                logging.error('Multiple entries for "%s" found. Accepting last entry.' % tokens[0])
                            key = _smart_lower(tokens[0], self.aggressive_lowercasing)
                            if key not in self:
                                self[key] = []
                            self[key].extend(to_insert)

                            # note- do not attempt to lowercase if the thesaurus
                            #  has not already been lowercased- may result in
                            # multiple neighbour lists for the same entry

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

    def to_sparse_matrix(self, row_transform=None):
        """
        Converts the vectors held in this object to a scipy sparse matrix
        :return: a tuple containing
            1) the sparse matrix, in which rows correspond to the order of this object's iteritems()
            2) a sorted list of all features (column labels of the matrix)
            3) a sorted list of all entries (row labels of the matrix)
        :rtype: tuple
        """
        from sklearn.feature_extraction import DictVectorizer
        import numpy as np

        self.v = DictVectorizer(sparse=True, dtype=np.int32)
        # todo unit test this! v.vocabulary must equal columns of self.export_to_dissect_sparse

        # order in which rows are iterated is not guaranteed if the dict if modified, but we're not doing that,
        # so it's all fine
        mat = self.v.fit_transform([dict(fv) for fv in self.itervalues()])
        rows = [k for k in self.iterkeys()]
        if row_transform:
            rows = map(row_transform, rows)

        return mat, self.v.feature_names_, rows

    def to_dissect_space(self):
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

# END OF CLASS
def _smart_lower(words_with_pos, aggressive_lowercasing=True):
    """
    Lowercase just the words and not their PoS tags
    """
    if not aggressive_lowercasing:
        return words_with_pos

    unigrams = words_with_pos.split(' ')
    words = []
    for unigram in unigrams:
        try:
            word, pos = unigram.split('/')
        except ValueError:
            # no pos
            word, pos = words_with_pos, ''

        words.append('/'.join([word.lower(), pos]) if pos else word.lower())

    return ' '.join(words)
