# coding=utf-8
from collections import defaultdict
import logging


def read_thesaurus(**kwargs):
    """
    reads from disk, must be invoked before get_thesaurus
    """
    global last_read_thesaurus
    last_read_thesaurus = Thesaurus(**kwargs)
    return last_read_thesaurus


def get_thesaurus():
    """
    returns the last thesaurus that was read
    NB! If you change the settings (e.g. last_read_thesaurus.k=1) you must
    also invoke read_thesaurus
    """
    return last_read_thesaurus


class Thesaurus(dict):
    def __init__(self, thesaurus_files='', sim_threshold=0, k=1,
                 include_self=False):
        self.thesaurus_files = thesaurus_files
        self.sim_threshold = sim_threshold
        self.k = k
        self.include_self = include_self
        self.preloaded_thesauri = {}

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
            logging.getLogger().warn("No thesaurus specified")

        result = {}
        for path in self.thesaurus_files:
            if path in self.preloaded_thesauri and self.use_cache:
                # logging.getLogger().info('Returning cached thesaurus '
                #                          'for %s' % path)
                result.update(self.preloaded_thesauri[path])
            else:
                logging.getLogger().info(
                    'Loading thesaurus %s from disk' % path)
                logging.getLogger().debug(
                    'threshold %r, k=%r' % (self.sim_threshold, self.k))

                FILTERED = '___FILTERED___'.lower()
                curr_thesaurus = defaultdict(list)
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
                                         _iterate_nonoverlapping_pairs(
                                             tokens, 1, self.k)
                                         if
                                         word != FILTERED and
                                         float(sim) > self.sim_threshold]
                            if self.include_self:
                                to_insert.insert(0, (tokens[0].lower(), 1.0))
                                # the step above may filter out all neighbours
                                # of an entry. if this happens, do not bother
                                # adding it
                            if len(to_insert) > 0:
                                if tokens[0] in curr_thesaurus:
                                    logging.getLogger().error(
                                        'Multiple entries for "%s" found' %
                                        tokens[0])
                                curr_thesaurus[tokens[0].lower()].extend(
                                    to_insert)

                # note- do not attempt to lowercase if the thesaurus has not
                # already been lowercased- may result in multiple neighbour lists
                # for the same entry
                logging.getLogger().info('Caching thesaurus %s' % path)
                self.preloaded_thesauri[path] = curr_thesaurus
                result.update(curr_thesaurus)

        # logging.getLogger().info(
        #     'Thesaurus contains %d entries' % len(result))
        # logging.getLogger().debug(
        #     'Thesaurus sample %r' % result.items()[:2])
        self.update(result)
        # END OF CLASS


def _iterate_nonoverlapping_pairs(iterable, beg, num_pairs):
    for i in xrange(beg, min(len(iterable) - 1, 2 * num_pairs),
                    2):  # step size 2
        yield (iterable[i], iterable[i + 1])
