from discoutils.tokens import DocumentFeature
from discoutils.thesaurus_loader import Thesaurus
from thesisgenerator.utils.reflection_utils import get_named_object


def get_token_handler(handler_name, k, transformer_name, thesaurus):
    # k- parameter for _paraphrase
    # sim_transformer- callable that transforms the raw sim scores in
    # _paraphrase
    # todo replace k with a named object
    handler = get_named_object(handler_name)
    transformer = get_named_object(transformer_name)
    return handler(k, transformer, thesaurus)


class BaseFeatureHandler():
    """
    Handles features the way standard Naive Bayes does:
        - in vocabulary, in thesaurus: only insert feature itself
        - IV,OOT: feature itself
        - OOV, IT: ignore feature
        - OOV, OOT: ignore feature
    """

    def __init__(self, *args):
        pass

    def handle_IV_IT_feature(self, **kwargs):
        self._insert_feature_only(**kwargs)

    def handle_IV_OOT_feature(self, **kwargs):
        self._insert_feature_only(**kwargs)

    def handle_OOV_IT_feature(self, **kwargs):
        self._ignore_feature(**kwargs)

    def handle_OOV_OOT_feature(self, **kwargs):
        self._ignore_feature(**kwargs)


    def _insert_feature_only(self, feature_index_in_vocab, j_indices, values, **kwargs):
        #logging.debug('Inserting feature in doc %d: %s', doc_id, feature)
        j_indices.append(feature_index_in_vocab)
        values.append(1)

    def _ignore_feature(self, doc_id, feature, **kwargs):
        #logging.debug('Ignoring feature in doc %d: %s', doc_id, feature)
        pass

    def _paraphrase(self, feature, vocabulary, j_indices, values, stats, **kwargs):
        """
        Replaces term with its k nearest neighbours from the thesaurus

        Parameters
        ----------
        neighbour_source : callable, returns a thesaurus-like object (a list of
          (neighbour, sim) tuples, sorted by highest sim first,
          acts as a defaultdict(list) ). The callable takes one parameter for
          compatibility purposes- one of the possible callables I want to
          use here requires access to the vocabulary.
           The default behaviour is to return a callable pointing to the
           currently loaded thesaurus.
        """

        # logging.debug('Paraphrasing %r in doc %d', feature, doc_id)
        neighbours = self.thesaurus.get_nearest_neighbours(feature)
        if isinstance(self.thesaurus, Thesaurus):
            # precomputed thesauri do not guarantee that the returned neighbours will be in vocabulary
            # these should by now only the used in testing though
            neighbours = [(neighbour, sim) for (neighbour, sim) in neighbours
                          if DocumentFeature.from_string(neighbour) in vocabulary]
        event = [feature.tokens_as_str(), len(neighbours)]
        for neighbour, sim in neighbours[:self.k]:
            # the document may already contain the feature we
            # are about to insert into it,
            # a merging strategy is required,
            # e.g. what do we do if the document has the word X
            # in it and we encounter X again. By default,
            # scipy uses addition
            df = DocumentFeature.from_string(neighbour)
            j_indices.append(vocabulary.get(df))
            values.append(self.sim_transformer(sim))
            # track the event
            event.extend([neighbour, sim])
        stats.register_paraphrase(tuple(event))


class SignifierSignifiedFeatureHandler(BaseFeatureHandler):
    """
    Handles features the way standard Naive Bayes does, except
        - OOV, IT: insert the first K IV neighbours from thesaurus instead of
        ignoring the feature
    """

    def __init__(self, k, sim_transformer, thesaurus):
        self.k = k
        self.sim_transformer = sim_transformer
        self.thesaurus = thesaurus

    def handle_OOV_IT_feature(self, **kwargs):
        self._paraphrase(**kwargs)


class SignifiedOnlyFeatureHandler(BaseFeatureHandler):
    """
    Ignores all OOT features and inserts the first K IV neighbours from
    thesaurus for all IT features
    """

    def __init__(self, k, sim_transformer, thesaurus):
        self.k = k
        self.sim_transformer = sim_transformer
        self.thesaurus = thesaurus

    def handle_OOV_IT_feature(self, **kwargs):
        self._paraphrase(**kwargs)

    handle_IV_IT_feature = handle_OOV_IT_feature

    def handle_IV_OOT_feature(self, **kwargs):
        self._ignore_feature(**kwargs)


class SignifierRandomBaselineFeatureHandler(SignifiedOnlyFeatureHandler):
    """
    Ignores all OOT features and inserts K random IV tokens for all IT features
    """

    def __init__(self, k, sim_transformer, thesaurus):
        self.k = k
        self.sim_transformer = sim_transformer
        self.thesaurus = thesaurus

    def handle_OOV_IT_feature(self, **kwargs):
        self._paraphrase(**kwargs)

    handle_IV_IT_feature = handle_OOV_IT_feature