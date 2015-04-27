import logging
from os.path import join, basename
from pprint import pprint
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.semantic_space.space import Space
from composes.composition.full_additive import FullAdditive
from composes.composition.lexical_function import LexicalFunction
from composes.utils import io_utils
from composes.utils.regression_learner import RidgeRegressionLearner, LstsqRegressionLearner
from discoutils.tokens import DocumentFeature
from discoutils.thesaurus_loader import Vectors
from discoutils.misc import mkdirs_if_not_exists


def _translate_byblo_to_dissect(events_file, row_transform=lambda x: x):
    """
    Translates Byblo-made vectors file to dissect format in the absence of features/entries files
    :param events_file: path to byblo-made vectors
    :type events_file: str
    :return: prefix of dissect-compatible data files
    :rtype: str
    """
    # remove duplicate head noun vectors, converting to a dissect sparse matrix format
    logging.info('Converting %s to DISSECT format', events_file)
    t = Vectors.from_tsv(events_file)
    t.to_dissect_sparse_files(events_file, row_transform=row_transform)


def train_baroni_guevara_composers(all_vectors,
                                   ROOT_DIR,
                                   baroni_output_path, guevara_output_path,
                                   baroni_threshold=10):
    SVD_DIMS = 100
    baroni_training_phrase_types = {'AN', 'NN'}  # what kind of NPs to train Baroni composer for

    # prepare the input files to be fed into Dissect
    mkdirs_if_not_exists(ROOT_DIR)

    filename = basename(all_vectors)
    noun_events_file = join(ROOT_DIR, '%s-onlyN-SVD%d.tmp' % (filename, SVD_DIMS))
    NPs_events_file = join(ROOT_DIR, '%s-onlyPhrases-SVD%d.tmp' % (filename, SVD_DIMS))

    thes = Vectors.from_tsv(all_vectors, lowercasing=False)
    thes.to_tsv(noun_events_file,
                entry_filter=lambda x: x.type == '1-GRAM' and x.tokens[0].pos == 'N')
    _translate_byblo_to_dissect(noun_events_file)

    thes.to_tsv(NPs_events_file,
                entry_filter=lambda x: x.type in baroni_training_phrase_types,
                row_transform=lambda x: x.replace(' ', '_'))
    _translate_byblo_to_dissect(NPs_events_file)

    my_space = Space.build(data="{}.sm".format(noun_events_file),
                           rows="{}.rows".format(noun_events_file),
                           cols="{}.cols".format(noun_events_file),
                           format="sm")
    logging.info('Each unigram vector has dimensionality %r', my_space.element_shape)

    # create a peripheral space
    my_per_space = PeripheralSpace.build(my_space,
                                         data="{}.sm".format(NPs_events_file),
                                         rows="{}.rows".format(NPs_events_file),
                                         # The columns of the peripheral space have to be identical to those
                                         # in the core space (including their order)!
                                         cols="{}.cols".format(NPs_events_file),
                                         format="sm")
    logging.info('Each phrase vector has dimensionality %r', my_per_space.element_shape)

    # use the model to compose words in my_space
    all_data = []
    for phrase in my_per_space._row2id:
        # make sure there are only NPs here
        if DocumentFeature.from_string(phrase.replace(' ', '_')).type in baroni_training_phrase_types:
            adj, noun = phrase.split('_')
            all_data.append((adj, noun, '%s_%s' % (adj, noun)))

    # train a composition model on the data and save it
    baroni = LexicalFunction(min_samples=baroni_threshold, learner=RidgeRegressionLearner())
    guevara = FullAdditive()
    for composer, out_path in zip([baroni, guevara],
                                  [baroni_output_path, guevara_output_path]):
        composer.train(all_data, my_space, my_per_space)
        io_utils.save(composer, out_path)
        logging.info('Saved trained composer to %s', out_path)


def train_grefenstette_multistep_composer(all_vectors_file, root_dir):
    # ex19.py
    # -------

    mkdirs_if_not_exists(root_dir)
    filename = basename(all_vectors_file)
    noun_events_file = join(root_dir, '%s-onlyN.tmp' % filename)
    verb_events_file = join(root_dir, '%s-onlyV.tmp' % filename)
    vo_events_file = join(root_dir, '%s-onlyVO.tmp' % filename)
    svo_events_file = join(root_dir, '%s-onlySVO.tmp' % filename)

    # this has unigrams and observed phrases
    thes = Vectors.from_tsv(all_vectors_file)
    # thes.to_tsv(noun_events_file,
    #             entry_filter=lambda x: x.type == '1-GRAM' and x.tokens[0].pos == 'N')
    # _translate_byblo_to_dissect(noun_events_file)
    # thes.to_tsv(verb_events_file,
    #             entry_filter=lambda x: x.type == '1-GRAM' and x.tokens[0].pos == 'V')
    # _translate_byblo_to_dissect(verb_events_file)
    # thes.to_tsv(vo_events_file,
    #             entry_filter=lambda x: x.type == 'VO')
    # _translate_byblo_to_dissect(vo_events_file)
    # thes.to_tsv(svo_events_file,
    #             entry_filter=lambda x: x.type == 'SVO')
    # _translate_byblo_to_dissect(svo_events_file)

    train_vo_data, train_v_data = [], []
    for phrase in thes.keys():
        df = DocumentFeature.from_string(phrase)
        if df.type == 'SVO':
            train_vo_data.append((df[1:], df[0], df))
        if df.type == 'VO':
            train_v_data.append((df[0], df[1], df))

    # pprint(train_vo_data)
    # pprint(train_v_data)

    # training data1: VO N -> SVO
    # train_vo_data = [("hate_boy", "man", "man_hate_boy"),
    # ("hate_man", "man", "man_hate_man"),
    #                  ("hate_boy", "boy", "boy_hate_boy"),
    #                  ("hate_man", "boy", "boy_hate_man")
    #                  ]

    # training data2: V N -> VO
    # train_v_data = [("hate", "man", "hate_man"),
    #                 ("hate", "boy", "hate_boy")
    #                 ]

    # load N and SVO spaces
    n_space = Space.build(data=noun_events_file + '.sm',
                          cols=noun_events_file + '.cols',
                          format="sm")

    svo_space = Space.build(data=svo_events_file + '.sm',
                            cols=svo_events_file + '.cols',
                            format="sm")

    print("\nInput SVO training space:")
    print(svo_space.id2row)
    # print(svo_space.cooccurrence_matrix)

    # 1. train a model to learn VO functions on train data: VO N -> SVO
    print("\nStep 1 training")
    vo_model = LexicalFunction(learner=LstsqRegressionLearner())  # todo ridge regr
    vo_model.train(train_vo_data, n_space, svo_space)

    # 2. train a model to learn V functions on train data: V N -> VO
    # where VO space: function space learned in step 1
    print("\nStep 2 training")
    vo_space = vo_model.function_space
    v_model = LexicalFunction(learner=LstsqRegressionLearner())  # todo ridge regr
    v_model.train(train_v_data, n_space, vo_space)

    # print the learned model
    print("\n3D Verb space")
    print(v_model.function_space.id2row)
    # print(v_model.function_space.cooccurrence_matrix)


    # 3. use the trained models to compose new SVO sentences

    # 3.1 use the V model to create new VO combinations
    vo_composed_space = v_model.compose([("hate", "woman", "hate_woman"),
                                         ("hate", "man", "hate_man")],
                                        n_space)

    # 3.2 the new VO combinations will be used as functions:
    # load the new VO combinations obtained through composition into
    # a new composition model
    expanded_vo_model = LexicalFunction(function_space=vo_composed_space,
                                        intercept=v_model._has_intercept)

    # 3.3 use the new VO combinations by composing them with subject nouns
    # in order to obtain new SVO sentences
    svo_composed_space = expanded_vo_model.compose([("hate_woman", "woman", "woman_hates_woman"),
                                                    ("hate_man", "man", "man_hates_man")],
                                                   n_space)

    # print the composed spaces:
    print("\nSVO composed space:")
    print(svo_composed_space.id2row)
    # print(svo_composed_space.cooccurrence_matrix)


if __name__ == '__main__':
    prefix = '/Volumes/LocalDataHD/m/mm/mmb28/NetBeansProjects/FeatureExtractionToolkit/'
    vect = 'exp10-13b/exp10-with-obs-phrases-SVD100.events.filtered.strings'
    train_grefenstette_multistep_composer(join(prefix, vect),
                                          join(prefix, 'gref_multistep'))