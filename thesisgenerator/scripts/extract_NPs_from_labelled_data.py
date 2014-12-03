from collections import Counter
import sys
import logging
from discoutils.tokens import DocumentFeature

sys.path.append('.')
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.utils.data_utils import get_tokenized_data, get_tokenizer_settings_from_conf, get_all_corpora
from thesisgenerator.utils.conf_file_utils import parse_config_file
import numpy as np


def get_all_NPs(path_to_existing='NPs_in_R2_MR_tech_am/r2-mr-technion-am-ANsNNs.txt',
                include_unigrams=False):
    """
    Finds all noun-noun and adj-noun compounds (and optionally adjs and nouns) in all labelled corpora
    mentioned in the conf files.
    :param path_to_existing: Path to the output of this when it was last ran. Can save lots of time.
    :param include_unigrams: if False, only NPs will be returned
    :rtype: set of DocumentFeature
    """
    accepted_df_types = {'AN', 'NN', '1-GRAM'} if include_unigrams else {'AN', 'NN'}
    if path_to_existing:
        logging.info('Returning precompiled list of NPs from %s', path_to_existing)
        result = set()
        with open(path_to_existing) as infile:
            for line in infile:
                df = DocumentFeature.from_string(line.strip())
                if df.type in accepted_df_types:
                    result.add(df)
            return result

    all_nps = set()
    for corpus_path, conf_file in get_all_corpora().items():
        logging.info('--------------------')
        logging.info('Processing corpus %s', corpus_path)
        conf, _ = parse_config_file(conf_file)
        x_tr, _, _, _ = get_tokenized_data(conf['training_data'] + '.gz',
                                           get_tokenizer_settings_from_conf(conf),
                                           test_data=conf['test_data'])
        assert not conf['test_data']
        assert len(x_tr) > 0
        logging.info('Documents in this corpus: %d', len(x_tr))

        # this will include vectors for all unigrams that occur in the labelled set in the output file
        # the composition code for other methods (in dump_all_composed_vectors.py) include vectors for all
        # unigrams that occur in the UNLABELLED set. This shouldn't make a difference as during classification we search for
        # neighbours of each doc feature among the features the occur in BOTH the labelled and unlabelled set
        vect = ThesaurusVectorizer(min_df=1, ngram_range=(1, 1),
                                   extract_SVO_features=False, extract_VO_features=False,
                                   unigram_feature_pos_tags=set('NJ'))
        data_matrix, voc = vect.fit_transform(x_tr, np.ones(len(x_tr)))
        logging.info('Found %d document features in this corpus', len(voc))
        all_nps |= set(foo for foo in voc.keys() if foo.type in accepted_df_types)  # set union
    logging.info('Found a total of %d features in all corpora', len(all_nps))
    logging.info('Their types are %r', Counter(df.type for df in all_nps))
    return all_nps


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")


    # (ROOT
    # (NP (NN acquisition) (NN pact)))
    #
    # (ROOT
    # (NP (JJ pacific) (NN stock)))
    stanford_NP_pattern = '(ROOT\n (NP ({} {}) ({} {})))\n\n'

    # (ROOT
    # (NP (NN roads)))
    # I checked that this extracts the neural word embedding for the word
    stanford_unigram_pattern = '(ROOT\n (NP ({} {})))\n\n'

    seen_modifiers = set()
    with open('r2-mr-technion-am-ANs-NNs-julie.txt', 'w') as outf_julie, \
            open('r2-mr-technion-am-ANs-NNs-socher.txt', 'w') as outf_socher, \
            open('r2-mr-technion-am-modifiers.txt', 'w') as outf_mods, \
            open('r2-mr-technion-am-ANsNNsUnigrams.txt', 'w') as outf_plain:
        all_nps = get_all_NPs(path_to_existing=False, include_unigrams=True)
        for item in all_nps:
            outf_plain.write(item.tokens_as_str())
            outf_plain.write('\n')
            if item.type in {'AN', 'NN'}:
                # write in my underscore-separated format
                first = str(item.tokens[0])
                second = str(item.tokens[1])

                # write just the modifier separately
                if first not in seen_modifiers:
                    outf_mods.write('%s\n' % first)
                    seen_modifiers.add(first)

                # write the phrase in Julie's format
                if item.type == 'AN':
                    string = '{}:amod-HEAD:{}\n'.format(first, second)
                else:
                    string = '{}:nn-DEP:{}\n'.format(second, first)
                outf_julie.write(string)

                # write the phrase in Socher's format
                string = stanford_NP_pattern.format(item.tokens[0].pos * 2, item.tokens[0].text,
                                                    item.tokens[1].pos * 2, item.tokens[1].text)
                outf_socher.write(string)
            elif item.type == '1-GRAM':
                string = stanford_unigram_pattern.format(item.tokens[0].pos * 2, item.tokens[0].text)
                outf_socher.write(string)
            else:  # there shouldn't be any other features
                raise ValueError('Item %r has the wrong feature type: %s' % (item, item.type))

    logging.info('Done')



