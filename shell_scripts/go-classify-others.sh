#!/bin/bash

# Parameters for Sun Grid Engine submition
# ============================================

# Shell to use
#$ -S /bin/bash

# All paths relative to current working directory
#$ -cwd

# List of queues
#$ -q nlp-amd,eng-inf_parallel.q,parallel.q,serial.q

# Define parallel environment for N cores
#$ -pe openmp 5

# Send mail to. (Comma separated list)
#$ -M mmb28@sussex.ac.uk

# When: [b]eginning, [e]nd, [a]borted and reschedules, [s]uspended, [n]one
#$ -m n

# Validation level (e = reject on all problems)
#$ -w e

# Merge stdout and stderr streams: yes/no
#$ -j yes

#$ -t 31-276
#$ -tc 70
python thesisgenerator/plugins/experimental_utils.py $SGE_TASK_ID 
