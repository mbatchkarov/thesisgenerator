# coding=utf-8
import logging
from os.path import basename


class Thesaurus(dict):
    def __init__(self, thesaurus_files='', sim_threshold=0, include_self=False):
        self.thesaurus_files = thesaurus_files
        self.thesaurus_names = map(basename, thesaurus_files)
        self.sim_threshold = sim_threshold
        self.include_self = include_self
        # make this class act like a defaultdict(list)
        #self.default_factory = list

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
            logging.info('Loading thesaurus %s from disk' % path)

            FILTERED = '___FILTERED___'.lower()
            with open(path) as infile:
                for line in infile:
                    tokens = line.strip().split('\t')
                    if len(tokens) % 2 == 0:
                    # must have an odd number of things, one for the entry
                    # and pairs for (neighbour, similarity)
                        continue
                    if tokens[0] != FILTERED:
                        to_insert = [(word.lower(), float(sim)) for
                                     (word, sim)
                                     in
                                     _iterate_nonoverlapping_pairs(tokens, 1)
                                     if
                                     word.lower() != FILTERED and
                                     float(sim) > self.sim_threshold]
                        if self.include_self:
                            to_insert.insert(0, (tokens[0].lower(), 1.0))
                            # the step above may filter out all neighbours
                            # of an entry. if this happens, do not bother
                            # adding it
                        if len(to_insert) > 0:
                            if tokens[0] in self:
                                logging.error('Multiple entries for "%s" '
                                              'found' % tokens[0])
                            key = tokens[0].lower()
                            if not self.has_key(key):
                                self[key] = []
                            self[key].extend(to_insert)

                            # note- do not attempt to lowercase if the thesaurus
                            #  has not already been lowercased- may result in
                            # multiple neighbour lists for the same entry


# END OF CLASS
def _iterate_nonoverlapping_pairs(iterable, beg):
    for i in xrange(beg, min(len(iterable) - 1, len(iterable)),
                    2):  # step size 2
        yield (iterable[i], iterable[i + 1])
