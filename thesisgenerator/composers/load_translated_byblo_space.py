import logging
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.semantic_space.space import Space
from composes.composition.full_additive import FullAdditive
from composes.composition.lexical_function import LexicalFunction
from composes.utils import io_utils
from discoutils.thesaurus_loader import Vectors


def _translate_byblo_to_dissect(events_file, row_transform=lambda x: x):
    """
    Translated Byblo-made vectors file to dissect format in the absence of features/entries files
    :param events_file: path to byblo-made vectors
    :type events_file: str
    :return: prefix of dissect-compatible data files
    :rtype: str
    """
    # remove duplicate head noun vectors, converting to a dissect sparse matrix format
    logging.info('Converting %s to DISSECT format', events_file)
    t = Vectors.from_tsv(events_file)
    output_file = '{}.uniq'.format(events_file)
    t.to_dissect_sparse_files(output_file, row_transform=row_transform)
    return output_file


def train_baroni_guevara_composers(noun_events_file, ANs_events_file,
                          baroni_output_path, guevara_output_path,
                          row_transform=lambda x: x, baroni_threshold=10):
    # prepare the input files to be fed into Dissect
    cleaned_nouns_file = _translate_byblo_to_dissect(noun_events_file, row_transform=row_transform)
    cleaned_an_file = _translate_byblo_to_dissect(ANs_events_file, row_transform=row_transform)

    my_space = Space.build(data="{}.sm".format(cleaned_nouns_file),
                           rows="{}.rows".format(cleaned_nouns_file),
                           cols="{}.cols".format(cleaned_nouns_file),
                           format="sm")
    logging.info('Each unigram vector has dimensionality %r', my_space.element_shape)

    # create a peripheral space
    my_per_space = PeripheralSpace.build(my_space,
                                         data="{}.sm".format(cleaned_an_file),
                                         rows="{}.rows".format(cleaned_an_file),
                                         # The columns of the peripheral space have to be identical to those
                                         # in the core space (including their order)!
                                         cols="{}.cols".format(cleaned_nouns_file),
                                         format="sm")
    logging.info('Each phrase vector has dimensionality %r', my_per_space.element_shape)

    # use the model to compose words in my_space
    all_data = []
    for an in my_per_space._row2id:
        adj, noun = an.split('_')
        all_data.append((adj, noun, '%s_%s' % (adj, noun)))


    # train a composition model on the data and save it
    for composer, out_path in zip([LexicalFunction(min_samples=baroni_threshold), FullAdditive()],
                                  [baroni_output_path, guevara_output_path]):
        composer.train(all_data, my_space, my_per_space)
        io_utils.save(composer, out_path)
        logging.info('Saved trained composer to %s', out_path)
