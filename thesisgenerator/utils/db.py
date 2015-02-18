import peewee as pw

db = pw.SqliteDatabase('db.sqlite')
db.connect()


class Vectors(pw.Model):
    algorithm = pw.CharField(null=False)  # how the *unigram* vectors were built, e.g. count_windows, Turian, word2vec
    dimensionality = pw.IntegerField(null=True)  # 0 to indicate no SVD was done, -1 for not applicable
    unlabelled_percentage = pw.FloatField(default=100., null=True)  # how much of the unlabelled data was used
    unlabelled = pw.CharField(null=True)  # path to unlabelled corpus that data was used, if I did it
    path = pw.CharField(null=True)  # where on disk the vectors are stored
    composer = pw.CharField()  # what composer was used to build phrasal vectors (if any)
    rep = pw.IntegerField(default=0)  # if the same vectors have been built multiple times, an explicit identifier
    use_ppmi = pw.BooleanField(default=0)

    modified = pw.DateField(null=True, default=None)  # when was the file last modifier
    size = pw.IntegerField(null=True, default=None)  # file size in MB
    gz_size = pw.IntegerField(null=True, default=None)  # file size of compressed JSON version

    class Meta:
        database = db

    def __str__(self):
        return 'Vectors: ' + ','.join(str(x) for x in [self.unlabelled, self.algorithm, self.composer,
                                                       self.dimensionality, self.rep, self.unlabelled_percentage])


class ClassificationExperiment(pw.Model):
    document_features = pw.CharField(default='AN_NN')  # AN+NN, AN only, NN only, ...
    use_similarity = pw.BooleanField(default=False)  # use phrase sim as pseudo term count
    use_random_neighbours = pw.BooleanField(default=False)
    decode_handler = pw.CharField(default='SignifiedOnlyFeatureHandler')  # signifier, signified, hybrid
    vectors = pw.ForeignKeyField(Vectors, null=True, default=None, on_delete='SET NULL')
    labelled = pw.CharField()  # name/path of labelled corpus used
    k = pw.IntegerField(default=3)  # how many neighbours entries are replaced with at decode time
    neighbour_strategy = pw.CharField(default='linear')  # how neighbours are found- linear or skipping strategy
    noise = pw.FloatField(default=0)

    date_ran = pw.DateField(null=True, default=None)
    git_hash = pw.CharField(null=True, default=None)

    class Meta:
        database = db

    def __str__(self):
        basic_settings = ','.join((str(x) for x in [self.labelled, self.vectors]))
        return '%s: %s' % (self.id, basic_settings)

    def get_important_fields(self):
        return

    def __eq__(self, other):
        if not isinstance(other, ClassificationExperiment):
            return False
        return (
            self.document_features == other.document_features and
            self.use_similarity == other.use_similarity and
            self.use_random_neighbours == other.use_random_neighbours and
            self.decode_handler == other.decode_handler and
            self.vectors.id == other.vectors.id and
            self.labelled == other.labelled and
            self.k == other.k and
            self.neighbour_strategy == other.neighbour_strategy and
            self.noise == other.noise
        )

    def __hash__(self):
        return hash((self.document_features, self.use_similarity,
                     self.use_random_neighbours, self.decode_handler,
                     self.labelled, self.k, self.neighbour_strategy,
                     self.noise) + (self.vectors.id if self.vectors else None, ))


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


class FullResults(pw.Model):
    id = pw.ForeignKeyField(ClassificationExperiment)
    classifier = pw.CharField(null=False)
    cv_fold = pw.IntegerField(null=False)
    accuracy_score = pw.DoubleField(null=False)
    macroavg_f1 = pw.DoubleField(null=False)
    microavg_f1 = pw.DoubleField(null=False)
    macroavg_rec = pw.DoubleField(null=False)
    microavg_rec = pw.DoubleField(null=False)
    microavg_prec = pw.DoubleField(null=False)
    macroavg_prec = pw.DoubleField(null=False)

    class Meta:
        database = db
        primary_key = pw.CompositeKey('id', 'cv_fold', 'classifier')


if __name__ == '__main__':
    print('Clearing database')
    Results.drop_table()
    FullResults.drop_table()
    ClassificationExperiment.drop_table()
    Vectors.drop_table()
    pw.create_model_tables([FullResults, Results, ClassificationExperiment, Vectors])