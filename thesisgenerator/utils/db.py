import peewee as pw
import socket
from discoutils.misc import is_gzipped, is_hdf, is_plaintext, Bunch

hostname = socket.gethostname()
if 'node' in hostname or 'apollo' in hostname:
    # keeping the DB on lustre is very slow, move to NFS
    dbpath = '/home/m/mm/mmb28/db.sqlite'
else:
    dbpath = 'db.sqlite'

db = pw.SqliteDatabase(dbpath)
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
    format = pw.CharField(null=True, default=None)  # how the file is stored: plaintext, gzip or hdf
    contents = pw.CharField(null=True, default=None)  # what vectors there are in the file, e.g. 1-GRAM, AN, NN,...

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # find out what format the file is in
        if self.path and self.size:  # check file exists
            if is_gzipped(self.path):
                self.format = 'gz'
            elif is_hdf(self.path):
                self.format = 'hdf'
            elif is_plaintext(self.path):
                self.format = 'txt'

        if isinstance(self.composer, Bunch):
            # check what entries are contained in this vector store
            self.contents = '+'.join(sorted(['1-GRAM', 'AN', 'NN', 'VO', 'SVO']))
            self.composer = self.composer.name
        if isinstance(self.composer, type):
            entry_types = self.composer.entry_types.union({'1-GRAM'})
            self.contents = '+'.join(sorted(entry_types))
            self.composer = self.composer.name

    class Meta:
        database = db

    def __str__(self):
        return 'Vectors: ' + ','.join(str(x) for x in [self.unlabelled, self.algorithm, self.composer,
                                                       self.dimensionality, self.rep, self.unlabelled_percentage])


class ClassificationExperiment(pw.Model):
    document_features_tr = pw.CharField(default='J+N+AN+NN')  # AN+NN, AN only, NN only, ...
    document_features_ev = pw.CharField(default='AN+NN')
    use_similarity = pw.BooleanField(default=False)  # use phrase sim as pseudo term count
    allow_overlap = pw.BooleanField(default=False)  # allow lexical overlap between features and its replacements
    use_random_neighbours = pw.BooleanField(default=False)
    decode_handler = pw.CharField(default='SignifiedOnlyFeatureHandler')  # signifier, signified, hybrid
    labelled = pw.CharField()  # name/path of labelled corpus used
    vectors = pw.ForeignKeyField(Vectors, null=True, default=None, on_delete='SET NULL', related_name='vectors')
    entries_of = pw.ForeignKeyField(Vectors, null=True, default=None, on_delete='SET NULL', related_name='entries_of')
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

    def __repr__(self):
        return str(self)

    def __key(self):
        x = (self.document_features_tr,
             self.document_features_ev,
             self.use_similarity,
             self.allow_overlap,
             self.use_random_neighbours,
             self.decode_handler,
             self.labelled,
             self.vectors.id if self.vectors else None,
             self.entries_of.id if self.entries_of else None,
             self.k,
             self.neighbour_strategy,
             self.noise,
             )
        return x

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())


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