import sys

sys.path.append('.')
import gzip, json
import networkx as nx
from networkx.readwrite.json_graph import node_link_data, node_link_graph
from discoutils.tokens import Token
from thesisgenerator.utils.data_utils import get_tokenized_data, get_all_corpora, get_tokenizer_settings_from_conf_file


def token_encode(t):
    if isinstance(t, Token):
        d = t.__dict__
        d.update({'__token__': True})
        return d
    raise TypeError


def token_decode(dct):
    d = dict(dct)
    if '__token__' in d:
        return Token(**d)
    else:
        return d


# t = json.dumps(Token('cat', 'N'), default=token_encode)
# print(t)
# t1 = json.loads(t, object_hook=token_decode, object_pairs_hook=token_decode)
# print(t1)


for corpus_path, conf_file in get_all_corpora().items():
    if 'movie' not in corpus_path:
        continue

    print(corpus_path)
    x_tr, y_tr, x_test, y_test = get_tokenized_data(corpus_path,
                                                    get_tokenizer_settings_from_conf_file(conf_file))
    # vect
    with gzip.open('%s.txt' % corpus_path, 'wb') as outfile:
        for document, label in zip(x_tr, y_tr):
            all_data = [label]
            for sent_parse_tree in document:
                data = node_link_data(sent_parse_tree)
                all_data.append(data)
            outfile.write(bytes(json.dumps(all_data, default=token_encode), 'UTF8'))
            outfile.write(bytes('\n', 'UTF8'))

with gzip.open('sample-data/movie-reviews-tagged.txt', 'rb') as infile:
    for line in infile:
        d = json.loads(line.decode('UTF8'), object_hook=token_decode)
        label = d[0]
        for js in d[1:]:
            g = node_link_graph(js)
            if len(g) == 0:
                print(js)
