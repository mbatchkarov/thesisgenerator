import sys
import logging

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from joblib import Memory
from thesisgenerator.scripts import dump_all_composed_vectors as dump
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.utils.data_utils import get_tokenized_data, get_tokenizer_settings_from_conf
from thesisgenerator.utils.conf_file_utils import parse_config_file
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

memory = Memory(cachedir='.', verbose=999)
get_cached_tokenized_data = memory.cache(get_tokenized_data, ignore=['*', '**'])
# get any conf file, tokenizer settings should be the same
conf, _ = parse_config_file('conf/exp1/exp1_base.conf')

all_text = []
for path in dump.all_classification_corpora:
    x_tr, _, _, _ = tokenised_data = get_cached_tokenized_data(conf['training_data'],
                                                               get_tokenizer_settings_from_conf(conf))
    assert len(x_tr) > 0
    all_text.extend(x_tr)
    logging.info('Documents so far: %d', len(all_text))

# this will include vectors for all unigrams that occur in the labelled set in the output file
# the composition code for other methods (in dump_all_composed_vectors.py) include vectors for all
# unigrams that occur in the UNLABELLED set. This shouldn't make a difference as during classification we search for
# neighbours of each doc feature among the features the occur in BOTH the labelled and unlabelled set
vect = ThesaurusVectorizer(min_df=1, ngram_range=(1, 1),
                           extract_SVO_features=False, extract_VO_features=False,
                           unigram_feature_pos_tags=set('NJ'))
data_matrix, voc = vect.fit_transform(all_text, np.ones(len(all_text)))
logging.info('Found a total of %d document features', len(voc))

# (ROOT
# (NP (NN acquisition) (NN pact)))
#
# (ROOT
# (NP (JJ pacific) (NN stock)))
stanford_NP_pattern = '(ROOT\n (NP ({} {}) ({} {})))\n\n'

# (ROOT
#   (NP (NN roads)))
# I checked that this extracts the neural word embedding for the word
stanford_unigram_pattern = '(ROOT\n (NP ({} {})))\n\n'

seen_modifiers = set()
with open('r2-mr-technion-am-ANs-NNs-julie.txt', 'w') as outf_julie, \
        open('r2-mr-technion-am-ANs-NNs-socher.txt', 'w') as outf_socher, \
        open('r2-mr-technion-am-modifiers.txt', 'w') as outf_mods:
    for item in list(voc.keys()):
        if item.type in {'AN', 'NN'}:
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



