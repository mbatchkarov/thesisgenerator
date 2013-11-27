import pytest
import networkx as nx

from thesisgenerator.plugins.bov import ThesaurusVectorizer
from thesisgenerator.plugins.tokenizers import XmlTokenizer, DocumentFeature, Token


@pytest.fixture
def vectorizer():
    return ThesaurusVectorizer()


@pytest.fixture
def black_cat_document_features(vectorizer):
    with open('thesisgenerator/resources/tokenizer/black_cat.tagged') as infile:
        txt = infile.read()
    t = XmlTokenizer()
    sentence, (parse_tree, token_index) = t.tokenize_doc(txt)[0]
    assert len(sentence) == len(parse_tree.nodes()) == 9
    return vectorizer.extract_features_from_dependency_tree(parse_tree, token_index)


@pytest.fixture
def empty_document_features(vectorizer):
    sentence, parse_tree, token_index = [], nx.DiGraph(), {}
    return vectorizer.extract_features_from_dependency_tree(parse_tree, token_index)


def test_extract_features_from_correct_dependency_tree(black_cat_document_features):
    print black_cat_document_features
    for adj, noun in [('big', 'cat'), ('black', 'cat'), ('small', 'bird'), ('gray', 'bird')]:
        f = DocumentFeature('AN', (Token(adj, 'J'), Token(noun, 'N')))
        assert f in black_cat_document_features

    assert DocumentFeature('VO', (Token('eat', 'V'), Token('bird', 'N') )) in black_cat_document_features
    assert DocumentFeature('SVO', (Token('cat', 'N'), Token('eat', 'V'), Token('bird', 'N') )) \
        in black_cat_document_features

# todo expand tests to check how feature extraction deals with incorrect parser output, e.g. amod between a N and a V
def test_extract_features_from_empty_dependency_tree(empty_document_features):
    assert not empty_document_features