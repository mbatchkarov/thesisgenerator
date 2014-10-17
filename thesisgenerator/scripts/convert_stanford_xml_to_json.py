import sys

sys.path.append('.')
import gzip, json
from networkx.readwrite.json_graph import node_link_graph
from discoutils.tokens import Token

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


with gzip.open('sample-data/movie-reviews-tagged.gz', 'rb') as infile:
    for line in infile:
        d = json.loads(line.decode('UTF8'), object_hook=token_decode)
        label = d[0]
        for js in d[1:]:
            g = node_link_graph(js)
            if len(g) == 0:
                print(js)
