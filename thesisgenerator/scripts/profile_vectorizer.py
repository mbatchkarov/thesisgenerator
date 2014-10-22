from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.plugins.stats import NoopStatsRecorder
from thesisgenerator.utils.conf_file_utils import parse_config_file
from thesisgenerator.utils.data_utils import (get_thesaurus, get_tokenized_data,
                                              get_tokenizer_settings_from_conf)

conf_file = 'conf/exp0/exp0_base.conf'
conf, _ = parse_config_file(conf_file)
thesaurus = get_thesaurus(conf)
vect = ThesaurusVectorizer(record_stats=False)
vect.thesaurus = thesaurus
vect.stats = NoopStatsRecorder()
x_tr, y_tr, x_test, y_test = get_tokenized_data('sample-data/amazon_grouped-tagged.gz',
                                                get_tokenizer_settings_from_conf(conf),
                                                gzip_json=True)

vocabulary, X = vect.fit_transform(x_tr, y_tr) # init internal state
print('--------  PASTE SHIT ABOVE IN IPYTHON ----------')
# vocabulary, X = vect._count_vocab(x_tr, False)
# %lprun -f vect.my_feature_extractor vect.my_feature_extractor(x_tr[0])
