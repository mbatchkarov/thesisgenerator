from glob import glob
import os

max_size = {'sample-data/amazon_grouped-tagged/Automotive': 28000,
            'sample-data/amazon_grouped-tagged/Baby': 5800,
            'sample-data/amazon_grouped-tagged/Beauty': 20000,
            'sample-data/amazon_grouped-tagged/Patio': 13000,
            'sample-data/amazon_grouped-tagged/Pet_Supplies': 13000}

for subdir in glob('sample-data/amazon_grouped-tagged/*'):
    print(subdir)
    sizes = sorted([(x, os.stat(x).st_size) for x in glob(subdir + '/*')],
                   key=lambda foo: foo[1])
    print(max_size[subdir])
    for file in sizes[max_size[subdir]:]:
        print(file)
        os.unlink(file[0])

