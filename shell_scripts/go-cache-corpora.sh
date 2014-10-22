#!/bin/bash

# Parameters for Sun Grid Engine submition
# ============================================

# Shell to use
#$ -S /bin/bash

# All paths relative to current working directory
#$ -cwd

# List of queues
#$ -q eng-inf_himem.q,parallel.q,eng-inf_parallel.q,nlp-amd

# Define parallel environment for N cores
#$ -pe openmp 17

# Send mail to. (Comma separated list)
#$ -M mmb28@sussex.ac.uk

# When: [b]eginning, [e]nd, [a]borted and reschedules, [s]uspended, [n]one
#$ -m eas

# Validation level (e = reject on all problems)
#$ -w e

# Merge stdout and stderr streams: yes/no
#$ -j yes
python thesisgenerator/scripts/cache_all_data.py --jobs 10 
