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
#$ -pe openmp 15

# Send mail to. (Comma separated list)
#$ -M mmb28@sussex.ac.uk

# When: [b]eginning, [e]nd, [a]borted and reschedules, [s]uspended, [n]one
#$ -m eas

# Validation level (e = reject on all problems)
#$ -w e

# Merge stdout and stderr streams: yes/no
#$ -j yes

#$ -N glove

python thesisgenerator/scripts/get_glove_vectors.py --stages reformat vectors compose
