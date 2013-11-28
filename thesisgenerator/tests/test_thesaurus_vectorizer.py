import pytest
import networkx as nx

from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.plugins.tokenizers import XmlTokenizer, DocumentFeature, Token


@pytest.fixture
def vectorizer():
    return ThesaurusVectorizer()


@pytest.fixture
def black_cat_parse_tree():
    with open('thesisgenerator/resources/tokenizer/black_cat.tagged') as infile:
        txt = infile.read()
    t = XmlTokenizer()
    sentence, (parse_tree, token_index) = t.tokenize_doc(txt)[0]
    assert len(sentence) == len(parse_tree.nodes()) == 9
    return parse_tree, token_index


def test_extract_features_from_correct_dependency_tree(black_cat_parse_tree, vectorizer):
    features = vectorizer.extract_features_from_dependency_tree(*black_cat_parse_tree)

    for adj, noun in [('big', 'cat'), ('black', 'cat'), ('small', 'bird'), ('gray', 'bird')]:
        f = DocumentFeature('AN', (Token(adj, 'J'), Token(noun, 'N')))
        assert f in features

    assert DocumentFeature('VO', (Token('eat', 'V'), Token('bird', 'N') )) in features
    assert DocumentFeature('SVO', (Token('cat', 'N'), Token('eat', 'V'), Token('bird', 'N') )) \
        in features


def test_extract_features_from_empty_dependency_tree(vectorizer):
    parse_tree, token_index = nx.DiGraph(), {}
    features = vectorizer.extract_features_from_dependency_tree(parse_tree, token_index)

    assert not features


@pytest.mark.parametrize(
    ('change_to', 'expected_feature_count'),
    [
        ('amod', 4),
        ('nsubj', 0),
        ('dobj', 2)
    ]
)
def test_extract_features_from_dependency_tree_with_wrong_relation_types(black_cat_parse_tree,
                                                                         vectorizer,
                                                                         change_to,
                                                                         expected_feature_count):
    parse_tree, token_index = black_cat_parse_tree

    #change all dependencies to amod
    for source, target, data in parse_tree.edges(data=True):
        if data['type'] != change_to:
            data['type'] = change_to

    for source, target, data in parse_tree.edges(data=True):
        print data

    features = vectorizer.extract_features_from_dependency_tree(parse_tree, token_index)
    # if all relations are changed to AMOD, we should get two adjective per noun
    # if all relations are changed to NSUBJ, we should get no AN/VO/SVO features as we're missing the required relations
    # to build and AN feature, we need an AMOD relation between a J and a N
    # to build and VO feature, we need an DOBJ relation between a V and a N
    # to build and AN feature, we need an DOBJ relation between a V and a N and a NSUBJ between the same V and another N
    assert len(features) == expected_feature_count

    #check that only amods between adj and nouns are extracted
    for adj, noun in [('big', 'cat'), ('black', 'cat'), ('small', 'bird'), ('gray', 'bird')]:
        f = DocumentFeature('AN', (Token(adj, 'J'), Token(noun, 'N')))
        assert (f in features) == (change_to == 'amod')

    ## check that no VO/SVO features are extracted
    feature = DocumentFeature('VO', (Token('eat', 'V'), Token('bird', 'N') ))
    assert (feature in features) == (change_to == 'dobj')

    feature = DocumentFeature('SVO', (Token('cat', 'N'), Token('eat', 'V'), Token('bird', 'N') ))
    assert feature not in features # we're missing the required relations to build SVO
