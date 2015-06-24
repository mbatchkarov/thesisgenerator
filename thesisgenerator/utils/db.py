import peewee as pw
import socket
from discoutils.misc import is_gzipped, is_hdf, is_plaintext

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
    unlabelled_percentage = pw.IntegerField(default=100, null=True)  # how much of the unlabelled data was used
    unlabelled = pw.CharField(null=True)  # path to unlabelled corpus that data was used, if I did it
    path = pw.CharField(null=True)  # where on disk the vectors are stored
    composer = pw.CharField()  # what composer was used to build phrasal vectors (if any)
    rep = pw.IntegerField(default=0)  # if the same vectors have been built multiple times, an explicit identifier

    modified = pw.DateField(null=True, default=None)  # when was the file last modifier
    size = pw.IntegerField(null=True, default=None)  # file size in MB
    format = pw.CharField(null=True, default=None)  # how the file is stored: plaintext, gzip or hdf

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

        if self.composer and not isinstance(self.composer, str):
            self.composer = self.composer.name

    class Meta:
        database = db

    def __str__(self):
        return 'Vectors: ' + ','.join(str(x) for x in [self.unlabelled, self.algorithm, self.composer,
                                                       self.dimensionality, self.rep, self.unlabelled_percentage])


class Expansions(pw.Model):
    vectors = pw.ForeignKeyField(Vectors, null=True, default=None, on_delete='SET NULL', related_name='vectors')
    entries_of = pw.ForeignKeyField(Vectors, null=True, default=None, on_delete='SET NULL', related_name='entries_of')
    allow_overlap = pw.BooleanField(default=False)  # allow lexical overlap between features and its replacements
    use_random_neighbours = pw.BooleanField(default=False)
    decode_handler = pw.CharField(default='SignifiedOnlyFeatureHandler')  # signifier, signified, hybrid
    k = pw.IntegerField(default=3)  # how many neighbours entries are replaced with at decode time
    noise = pw.FloatField(default=0)
    use_similarity = pw.BooleanField(default=False)  # use phrase sim as pseudo term count
    neighbour_strategy = pw.CharField(default='linear')  # how neighbours are found- linear or skipping strategy

    class Meta:
        database = db

    def _key(self):
        return (self.use_similarity,
                self.allow_overlap,
                self.use_random_neighbours,
                self.decode_handler,
                self.vectors.id if self.vectors else None,
                self.entries_of.id if self.entries_of else None,
                self.k,
                self.neighbour_strategy,
                self.noise)

    def __eq__(x, y):
        return x._key() == y._key()

    def __hash__(self):
        return hash(self._key())


class Clusters(pw.Model):
    num_clusters = pw.IntegerField(null=True, default=None)
    path = pw.CharField(null=True, default=None)
    # vectors must be consistent with Experiment.vectors
    vectors = pw.ForeignKeyField(Vectors, null=True, default=None, on_delete='SET NULL')

    class Meta:
        database = db

    def __str__(self):
        return 'CL:%d-vec%d' % (self.num_clusters, self.vectors.id)

    def _key(self):
        return (self.vectors.id,
                self.num_clusters,
                self.path)

    def __eq__(x, y):
        return x._key() == y._key()

    def __hash__(self):
        return hash(self._key())


class ClassificationExperiment(pw.Model):
    document_features_tr = pw.CharField(default='J+N+AN+NN')  # AN+NN, AN only, NN only, ...
    document_features_ev = pw.CharField(default='AN+NN')
    labelled = pw.CharField()  # name/path of labelled corpus used
    clusters = pw.ForeignKeyField(Clusters, null=True, default=None, on_delete='SET NULL',
                                  related_name='clusters')
    expansions = pw.ForeignKeyField(Expansions, null=True, default=None, on_delete='SET NULL',
                                    related_name='expansions')

    date_ran = pw.DateField(null=True, default=None)
    git_hash = pw.CharField(null=True, default=None)

    class Meta:
        database = db

    def __str__(self):
        if self.expansions:
            basic_settings = ','.join((str(x) for x in [self.labelled, self.expansions.vectors]))
        else:
            basic_settings = ','.join((str(x) for x in [self.labelled, self.clusters]))
        return '%s: %s' % (self.id, basic_settings)

    def __repr__(self):
        return str(self)

    def _key(self):
        key = (self.document_features_tr,
               self.document_features_ev,
               self.labelled,
               self.expansions.id if self.expansions else None,
               self.clusters.id if self.clusters else None)
        return key

    def __eq__(x, y):
        return x._key() == y._key()

    def __hash__(self):
        return hash(self._key())


class Results(pw.Model):
    id = pw.ForeignKeyField(ClassificationExperiment)
    classifier = pw.CharField(null=False)
    accuracy_mean = pw.DoubleField(null=False)
    accuracy_std = pw.DoubleField(null=False)
    microf1_mean = pw.DoubleField(null=False)
    microf1_std = pw.DoubleField(null=False)
    macrof1_mean = pw.DoubleField(null=False)
    macrof1_std = pw.DoubleField(null=False)

    _predictions = pw.BlobField(null=True)
    _gold = pw.BlobField(null=True)

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
    tables = [Vectors, Clusters, Expansions, ClassificationExperiment, FullResults, Results]
    pw.drop_model_tables(tables)
    pw.create_model_tables(tables)
    print('Tables are now', db.get_tables())
