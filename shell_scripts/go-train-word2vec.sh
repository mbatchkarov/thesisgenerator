#!/bin/bash

# Parameters for Sun Grid Engine submition
# ============================================

# Shell to use
#$ -S /bin/bash

# All paths relative to current working directory
#$ -cwd

# List of queues
#$ -q nlp-amd,inf.q,serial.q,eng-inf_himem.q,eng-inf_parallel.q

# Define parallel environment for N cores
#$ -pe openmp 10

# Send mail to. (Comma separated list)
#$ -M mmb28@sussex.ac.uk

# When: [b]eginning, [e]nd, [a]borted and reschedules, [s]uspended, [n]one
#$ -m eas

# Validation level (e = reject on all problems)
#$ -w e

# Merge stdout and stderr streams: yes/no
#$ -j yes

#$ -N word2vec

#$ -t 10-100:10
#$ -tc 10
python thesisgenerator/scripts/get_word2vec_vectors.py --stages vectors compose --percent $SGE_TASK_ID
