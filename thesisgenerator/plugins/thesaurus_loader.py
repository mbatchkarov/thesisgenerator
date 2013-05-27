# coding=utf-8
from collections import defaultdict
import logging

__author__ = 'mmb28'

preloaded_thesauri = {}
use_cache = True    # used to disable caching for testing purposes


def get_all_thesauri():
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
    global thesaurus_files, sim_threshold, k, include_self

    if not thesaurus_files:
        logging.getLogger().warn("No thesaurus specified")

    result = {}
    logging.getLogger().debug(thesaurus_files)
    for path in thesaurus_files:
        if path in preloaded_thesauri and use_cache:
            # logging.getLogger().info('Returning cached thesaurus '
            #                          'for %s' % path)
            result.update(preloaded_thesauri[path])
        else:
            logging.getLogger().info(
                'Loading thesaurus %s from disk' % path)
            logging.getLogger().debug(
                'threshold %r, k=%r' % (sim_threshold, k))

            FILTERED = '___FILTERED___'.lower()
            curr_thesaurus = defaultdict(list)
            with open(path) as infile:
                for line in infile:
                    tokens = line.strip().split('\t')
                    if len(tokens) % 2 == 0:
                    #must have an odd number of things, one for the entry and
                    # pairs for (neighbour, similarity)
                        continue
                    if tokens[0] != FILTERED:
                        to_insert = [(word.lower(), float(sim)) for
                                     (word, sim)
                                     in
                                     _iterate_nonoverlapping_pairs(
                                         tokens, 1, k)
                                     if
                                     word != FILTERED and
                                     float(sim) > sim_threshold]
                        if include_self:
                            to_insert.insert(0, (tokens[0].lower(), 1.0))
                            # the step above may filter out all neighbours of an
                        # entry. if this happens, do not bother adding it
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
            if use_cache:
                logging.getLogger().info('Caching thesaurus %s' % path)
                preloaded_thesauri[path] = curr_thesaurus
            result.update(curr_thesaurus)

    # logging.getLogger().info(
    #     'Thesaurus contains %d entries' % len(result))
    # logging.getLogger().debug(
    #     'Thesaurus sample %r' % result.items()[:2])
    return result


def _iterate_nonoverlapping_pairs(iterable, beg, num_pairs):
    for i in xrange(beg, min(len(iterable) - 1, 2 * num_pairs),
                    2):  # step size 2
        yield (iterable[i], iterable[i + 1])
