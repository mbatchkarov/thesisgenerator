from thesisgenerator.scripts import dump_all_composed_vectors as dump
from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.utils.data_utils import load_text_data_into_memory, load_tokenizer, tokenize_data
import logging
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s\t%(module)s.%(funcName)s ""(line %(lineno)d)\t%(levelname)s : %(""message)s")

all_text = []
for path in dump.all_classification_corpora:
    raw_data, data_ids = load_text_data_into_memory(path)

    tokenizer = load_tokenizer(
        joblib_caching=True,
        normalise_entities=False,
        use_pos=True,
        coarse_pos=True,
        lemmatize=True,
        lowercase=True,
        remove_stopwords=True,
        remove_short_words=False)

    x_tr, _, x_ev, _ = tokenize_data(raw_data, tokenizer, data_ids)
    if x_tr:
        all_text.extend(x_tr)
    if x_ev:
        all_text.extend(x_ev)
    print('Documents so far', len(all_text))

vect = ThesaurusVectorizer(min_df=1, ngram_range=(1, 1), extract_SVO_features=False, extract_VO_features=False,
                           unigram_feature_pos_tags=set('NJ'))
data_matrix, voc = vect.fit_transform(all_text, np.ones(len(all_text)))

# (ROOT
#   (NP (NN acquisition) (NN pact)))
#
# (ROOT
#   (NP (JJ pacific) (NN stock)))
stanford_NP_pattern = '(ROOT\n (NP ({} {}) ({} {})))\n\n'

# (ROOT
#   (NP (NN roads)))
# I checked that this extracts the neural word embedding for the word
stanford_unigram_pattern = '(ROOT\n (NP ({} {})))\n\n'

seen_modifiers = set()
with open('r2-mr-ANs-NNs-julie.txt', 'w') as outf_julie, \
        open('r2-mr-ANs-NNs-socher.txt', 'w') as outf_socher, \
        open('r2-mr-modifiers.txt', 'w') as outf_mods:
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





