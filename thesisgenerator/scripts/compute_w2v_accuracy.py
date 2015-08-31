import logging

from numpy import dot, float32 as REAL, array, argsort, ndarray

from discoutils.thesaurus_loader import Vectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc

from six import string_types


def log_accuracy(section):
    correct, incorrect = len(section['correct']), len(section['incorrect'])
    if correct + incorrect > 0:
        logging.info("%s: %.1f%% (%i/%i)" %
                     (section['section'], 100.0 * correct / (correct + incorrect),
                      correct, correct + incorrect))


def most_similar(vectors, positive=[], negative=[], topn=10):
    """
    Find the top-N most similar words. Positive words contribute positively towards the
    similarity, negative words negatively.

    This method computes cosine similarity between a simple mean of the projection
    weight vectors of the given words, and corresponds to the `word-analogy` and
    `distance` scripts in the original word2vec implementation.

    Example::

      >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
      [('queen', 0.50882536), ...]

    """
    if isinstance(positive, string_types) and not negative:
        # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
        positive = [positive]

    # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
    positive = [(word, 1.0) if isinstance(word, string_types + (ndarray,))
                else word for word in positive]
    negative = [(word, -1.0) if isinstance(word, string_types + (ndarray,))
                else word for word in negative]

    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive + negative:
        if isinstance(word, ndarray):
            mean.append(weight * word)
        elif word in vectors.vocab:
            mean.append(weight * vectors.get_vector(word).A.ravel())
            all_words.add(vectors.vocab[word])
        else:
            raise KeyError("word '%s' not in vocabulary" % word)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

    dists = dot(vectors.matrix, mean)
    if not topn:
        return dists
    best = argsort(dists)[::-1][:topn + len(all_words)]
    # ignore (don't return) words from the input
    result = [(vectors.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
    return result[:topn]


def accuracy(vectors, questions, most_similar=most_similar):
    """
    Compute accuracy of the model. `questions` is a filename where lines are
    4-tuples of words, split into sections by ": SECTION NAME" lines.
    See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

    The accuracy is reported (=printed to log and returned as a list) for each
    section separately, plus there's one aggregate summary at the end.

    Use `restrict_vocab` to ignore all questions containing a word whose frequency
    is not in the top-N most frequent words (default top 30,000).

    This method corresponds to the `compute-accuracy` script of the original C word2vec.

    """
    vectors.init_sims()

    ok_vocab = {v: k for k, v in dict(enumerate(v.df.index)).items()}
    index2word = dict(enumerate(vectors.df.index))
    ok_index = set(ok_vocab.values())
    vectors.vocab = ok_vocab

    sections, section = [], None
    for line_no, line in enumerate(utils.smart_open(questions)):
        # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
        line = utils.to_unicode(line)
        if line.startswith(': '):
            # a new section starts => store the old section
            if section:
                sections.append(section)
                log_accuracy(section)
            section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
        else:
            if not section:
                raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
            try:
                a, b, c, expected = [word.lower() for word in
                                     line.split()]  # TODO assumes vocabulary preprocessing uses lowercase, too...
            except:
                logging.info("skipping invalid line #%i in %s" % (line_no, questions))
            if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                logging.debug("skipping line #%i with OOV words: %s" % (line_no, line.strip()))
                continue

            ignore = set(ok_vocab[v] for v in [a, b, c])  # indexes of words to ignore
            predicted = None
            # find the most likely prediction, ignoring OOV words and input words
            for index in argsort(most_similar(vectors, positive=[b, c], negative=[a], topn=False))[::-1]:
                if index in ok_index and index not in ignore:
                    predicted = index2word[index]
                    if predicted != expected:
                        logging.debug("%s: expected %s, predicted %s" % (line.strip(), expected, predicted))
                    break
            if predicted == expected:
                section['correct'].append((a, b, c, expected))
            else:
                section['incorrect'].append((a, b, c, expected))
    if section:
        # store the last section, too
        sections.append(section)
        log_accuracy(section)

    total = {
        'section': 'total',
        'correct': sum((s['correct'] for s in sections), []),
        'incorrect': sum((s['incorrect'] for s in sections), []),
    }
    log_accuracy(total)
    sections.append(total)
    return sections


path = '../FeatureExtractionToolkit/word2vec_vectors/word2vec-wiki-nopos-100perc.unigr.strings.rep0'
v = Vectors.from_tsv(path)
accuracy(v, '../ExpLosion/questions-words.txt')
