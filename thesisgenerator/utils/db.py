from configobj import ConfigObj
import peewee as pw


config = ConfigObj('thesisgenerator/db-credentials')
settings = {'user': config['user'],
            'passwd': config['pass'],
            'host': config['server'],
            'port': 3306}

db = pw.MySQLDatabase(config['db'], **settings)
db.connect()


class Vectors(pw.Model):
    algorithm = pw.CharField(null=False)  # how the *unigram* vectors were built, e.g. count_windows, Turian, word2vec
    can_build = pw.BooleanField()  # whether I built the vectors myself or downloaded off the internet
    dimensionality = pw.IntegerField(null=True)  # 0 to indicate no SVD was done, -1 for not applicable
    unlabelled_percentage = pw.IntegerField(default=100, null=True)  # how much of the unlabelled data was used
    unlabelled = pw.CharField(null=True)  # path to unlabelled corpus that data was used, if I did it
    path = pw.CharField(null=True)  # where on disk the vectors are stored
    composer = pw.CharField()  # what composer was used to build phrasal vectors (if any)

    modified = pw.DateField(null=True, default=None)  # when was the file last modifier
    size = pw.IntegerField(null=True, default=None)  # file size in MB

    class Meta:
        database = db

    def __str__(self):
        return 'Vectors: ' + ','.join(str(x) for x in [self.algorithm, self.composer, self.dimensionality])


class ClassificationExperiment(pw.Model):
    document_features = pw.CharField(default='AN_NN')  # AN+NN, AN only, NN only, ...
    use_similarity = pw.BooleanField(default=False)  # use phrase sim as pseudo term count
    use_random_neighbours = pw.BooleanField(default=False)
    decode_handler = pw.CharField(default='SignifiedOnlyFeatureHandler')  # signifier, signified, hybrid
    vectors = pw.ForeignKeyField(Vectors, null=True, default=None, on_delete='SET NULL')
    labelled = pw.CharField()  # name/path of labelled corpus used

    date_ran = pw.DateField(null=True, default=None)
    git_hash = pw.CharField(null=True, default=None)

    class Meta:
        database = db

    def __str__(self):
        basic_settings = ','.join((str(x) for x in [self.labelled, self.vectors]))
        return '%s: %s' % (self.id, basic_settings)


class Results(pw.Model):
    id = pw.ForeignKeyField(ClassificationExperiment)
    classifier = pw.CharField(null=False)
    accuracy_mean = pw.DoubleField(null=False)
    accuracy_std = pw.DoubleField(null=False)
    microf1_mean = pw.DoubleField(null=False)
    microf1_std = pw.DoubleField(null=False)
    macrof1_mean = pw.DoubleField(null=False)
    macrof1_std = pw.DoubleField(null=False)

    class Meta:
        database = db
        primary_key = pw.CompositeKey('id', 'classifier')


if __name__ == '__main__':
    Results.drop_table()
    ClassificationExperiment.drop_table()
    Vectors.drop_table()
    pw.create_model_tables([Results, ClassificationExperiment, Vectors])

    # print(1)
    # v = Vectors.create(can_build=False, dimensionality=100, unlabelled='wtf')
    # ClassificationExperiment.create(document_features='AN', use_similarity=False, use_random_neighbours=False,
    # decode_handler='Hybrid', vectors=v)
    # print(Vectors.select().where(Vectors.unlabelled=='wtf').get())