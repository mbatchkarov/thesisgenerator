from collections import Counter
import sys
import logging
from discoutils.tokens import DocumentFeature
from discoutils.misc import mkdirs_if_not_exists

sys.path.append('.')
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.utils.data_utils import get_tokenized_data, get_tokenizer_settings_from_conf, get_all_corpora
from thesisgenerator.utils.conf_file_utils import parse_config_file
import numpy as np

ROOT = 'features_in_labelled'
ALL_FEATURES_FILE = '%s/all_features.txt' % ROOT
NP_MODIFIERS_FILE = '%s/np_modifiers.txt' % ROOT
VERBS_FILE = '%s/verbs.txt' % ROOT
SOCHER_FILE = '%s/socher.txt' % ROOT


def get_all_NPs_VPs(path_to_existing=ALL_FEATURES_FILE, include_unigrams=False, return_counts=False):
    """
    Finds all noun-noun and adj-noun compounds (and optionally adjs and nouns) in all labelled corpora
    mentioned in the conf files.
    :param path_to_existing: Path to the output of this when it was last ran. Can save lots of time.
    :param include_unigrams: if False, only NPs will be returned
    :rtype: set of DocumentFeature
    """
    accepted_df_types = {'AN', 'NN', 'VO', 'SVO', '1-GRAM'} if include_unigrams else {'AN', 'NN', 'VO', 'SVO'}
    if path_to_existing:
        logging.info('Returning precompiled list of NPs from %s', path_to_existing)
        result = dict()
        with open(path_to_existing) as infile:
            for line in infile:
                df, count = line.strip().split()
                df = DocumentFeature.from_string(df)
                if df.type in accepted_df_types:
                    result[df] = int(count)
            return result if return_counts else set(result.keys())

    all_nps = Counter()
    conf, _ = parse_config_file('conf/exp1-superbase.conf')
    for corpus_path in get_all_corpora():
        logging.info('--------------------')
        logging.info('Processing corpus %s', corpus_path)
        tok = get_tokenizer_settings_from_conf(conf)
        x_tr, _, _, _ = get_tokenized_data(corpus_path + '.gz', tok,
                                           test_data=conf['test_data'])
        assert not conf['test_data']
        assert len(x_tr) > 0
        logging.info('Documents in this corpus: %d', len(x_tr))

        # this will include vectors for all unigrams that occur in the labelled set in the output file
        # the composition code for other methods (in vectorstore.py) include vectors for all
        # unigrams that occur in the UNLABELLED set. This shouldn't make a difference as during classification we search for
        # neighbours of each doc feature among the features the occur in BOTH the labelled and unlabelled set
        vect = ThesaurusVectorizer(min_df=1,
                                   train_time_opts={'extract_unigram_features': set('JNV'),
                                                    'extract_phrase_features': set(['AN', 'NN', 'VO', 'SVO'])})
        data_matrix, voc = vect.fit_transform(x_tr, np.ones(len(x_tr)))
        feature_counts = np.array(data_matrix.sum(axis=0)).ravel()
        logging.info('Found %d document features in this corpus', len(voc))
        for df, idx in voc.items():
            if df.type in accepted_df_types:
                all_nps[df] += feature_counts[idx]
    logging.info('Found a total of %d features in all corpora', len(all_nps))
    logging.info('Their types are %r', Counter(df.type for df in all_nps.keys()))
    logging.info('PoS tags of unigrams are are %r',
                 Counter(df.tokens[0].pos for df in all_nps.keys() if df.type == '1-GRAM'))
    return all_nps


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

    # How stanford parser formats NPs and VPs
    # (ROOT
    # (NP (NN acquisition) (NN pact)))
    #
    # (ROOT
    # (NP (JJ pacific) (NN stock)))
    stanford_NP_pattern = '(ROOT\n (NP ({} {}) ({} {})))\n\n'

    # (ROOT
    # (S
    # (NP (NNS cats))
    # (VP (VBP eat)
    # (NP (NNS dogs)))))
    stanford_SVO_pattern = '(ROOT\n  (S\n    (NP (NN {}))\n    (VP (VB {})\n      (NP (NN {})))))\n\n'

    # (ROOT
    # (S
    # (VP (VB eat)
    # (NP (NNS cats)))))
    stanford_VO_pattern = '(ROOT\n  (S\n    (VP (VB {})\n      (NP (NN {})))))\n\n'

    # (ROOT
    # (NP (NN roads)))
    # I checked that this extracts the neural word embedding for the word
    stanford_unigram_pattern = '(ROOT\n (NP ({} {})))\n\n'

    mkdirs_if_not_exists(ROOT)
    logging.info('Writing all document features to files')
    seen_modifiers, seen_verbs = set(), set()
    with open(SOCHER_FILE, 'w') as outf_socher, \
            open(NP_MODIFIERS_FILE, 'w') as outf_mods, \
            open(VERBS_FILE, 'w') as outf_verbs, \
            open(ALL_FEATURES_FILE, 'w') as outf_plain:
        all_phrases = get_all_NPs_VPs(path_to_existing=False, include_unigrams=True)
        for item, count in all_phrases.items():
            # write in my underscore-separated format
            outf_plain.write('%s\t%s' % (str(item), int(count)))
            outf_plain.write('\n')

            if item.type in {'AN', 'NN'}:
                # write the phrase in Socher's format
                string = stanford_NP_pattern.format(item.tokens[0].pos * 2, item.tokens[0].text,
                                                    item.tokens[1].pos * 2, item.tokens[1].text)
                outf_socher.write(string)

            if item.type in {'VO', 'SVO'}:
                verb = str(item.tokens[-2])
                if verb not in seen_verbs:
                    seen_verbs.add(verb)
                    outf_verbs.write(verb)
                    outf_verbs.write('\n')

            if item.type == 'VO':
                string = stanford_VO_pattern.format(*[x.tokens[0].text for x in item])
                outf_socher.write(string)

            if item.type == 'SVO':
                string = stanford_SVO_pattern.format(*[x.tokens[0].text for x in item])
                outf_socher.write(string)

            if item.type in {'AN', 'NN'}:
                # write just the modifier separately
                first = str(item.tokens[0])
                second = str(item.tokens[1])
                if first not in seen_modifiers:
                    outf_mods.write('%s\n' % first)
                    seen_modifiers.add(first)

            if item.type == '1-GRAM':
                string = stanford_unigram_pattern.format(item.tokens[0].pos * 2, item.tokens[0].text)
                outf_socher.write(string)

            if item.type not in {'1-GRAM', 'AN', 'NN', 'VO', 'SVO'}:  # there shouldn't be any other features
                raise ValueError('Item %r has the wrong feature type: %s' % (item, item.type))

    logging.info('Done')
