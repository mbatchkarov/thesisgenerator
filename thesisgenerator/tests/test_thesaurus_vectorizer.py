import pytest
import networkx as nx

from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.plugins.tokenizers import XmlTokenizer
from thesisgenerator.plugins.tokens import DocumentFeature, Token


@pytest.fixture(scope='module')
def valid_AN_features():
    return [('big', 'cat'), ('black', 'cat'), ('small', 'bird'), ('red', 'bird')]


@pytest.fixture(scope='module')
def vectorizer():
    return ThesaurusVectorizer()


@pytest.fixture(scope='module')
def black_cat_parse_tree():
    with open('thesisgenerator/resources/tokenizer/black_cat.tagged') as infile:
        txt = infile.read()
    t = XmlTokenizer()
    sentence, (parse_tree, token_index) = t.tokenize_doc(txt)[0]
    assert len(sentence) == len(parse_tree.nodes()) == 14
    return parse_tree, token_index


@pytest.skip('Some dependency features manually disabled for performance reasons')
def test_extract_features_from_correct_dependency_tree(black_cat_parse_tree, vectorizer, valid_AN_features):
    features = vectorizer.extract_features_from_dependency_tree(*black_cat_parse_tree)

    for adj, noun in valid_AN_features:
        f = DocumentFeature('AN', (Token(adj, 'J'), Token(noun, 'N')))
        assert f in features

    assert DocumentFeature('VO', (Token('eat', 'V'), Token('bird', 'N') )) in features
    assert DocumentFeature('SVO', (Token('cat', 'N'), Token('eat', 'V'), Token('bird', 'N') )) \
        in features

    assert DocumentFeature('NN', (Token('heart', 'N'), Token('surgery', 'N') )) in features


def test_extract_features_from_empty_dependency_tree(vectorizer):
    parse_tree, token_index = nx.DiGraph(), {}
    features = vectorizer.extract_features_from_dependency_tree(parse_tree, token_index)

    assert not features


@pytest.skip('Some dependency features manually disabled for performance reasons')
@pytest.mark.parametrize(
    ('change_to', 'expected_feature_count'),
    [
        ('amod', 4),
        ('nsubj', 0),
        ('dobj', 4)
    ]
)
def test_extract_features_from_dependency_tree_with_wrong_relation_types(black_cat_parse_tree,
                                                                         vectorizer,
                                                                         change_to,
                                                                         expected_feature_count,
                                                                         valid_AN_features):
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
    for adj, noun in valid_AN_features:
        f = DocumentFeature('AN', (Token(adj, 'J'), Token(noun, 'N')))
        assert (f in features) == (change_to == 'amod')

    ## check that no VO/SVO features are extracted
    feature = DocumentFeature('VO', (Token('eat', 'V'), Token('bird', 'N') ))
    assert (feature in features) == (change_to == 'dobj')

    feature = DocumentFeature('SVO', (Token('cat', 'N'), Token('eat', 'V'), Token('bird', 'N') ))
    assert feature not in features # we're missing the required relations to build SVO
