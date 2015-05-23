#!/bin/bash

# Parameters for Sun Grid Engine submition
# ============================================

# Shell to use
#$ -S /bin/bash

# All paths relative to current working directory
#$ -cwd

# List of queues
#$ -q nlp-amd,eng-inf_parallel.q,parallel.q

# Define parallel environment for N cores
#$ -pe openmp 4

# Send mail to. (Comma separated list)
#$ -M mmb28@sussex.ac.uk

# When: [b]eginning, [e]nd, [a]borted and reschedules, [s]uspended, [n]one
#$ -m eas

# Validation level (e = reject on all problems)
#$ -w e

# Merge stdout and stderr streams: yes/no
#$ -j yes
#$ -N cluster

#$ -t 1-30
#$ -tc 15

SETTINGS_FILE=mmb28_cluster_settings.tmp.$RANDOM # python job below is invoked for each task in the jobs array, write to a separate file to avoid conflicts
python -c "
from os.path import splitext
from thesisgenerator.utils import db
for e in db.Clusters.select():
    root, ext = splitext(e.path)
    print(root, e.path, e.num_clusters)
" > $SETTINGS_FILE

SEED=$(awk "NR==$SGE_TASK_ID" $SETTINGS_FILE) 
echo $SEED
python thesisgenerator/plugins/kmeans_disco.py $SEED