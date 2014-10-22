#!/bin/bash

# Parameters for Sun Grid Engine submition
# ============================================

# Shell to use
#$ -S /bin/bash

# All paths relative to current working directory
#$ -cwd

# List of queues
#$ -q nlp-amd,eng-inf_himem.q,inf.q,serial.q

# Define parallel environment for N cores
#$ -pe openmp 20

# Send mail to. (Comma separated list)
#$ -M mmb28@sussex.ac.uk

# When: [b]eginning, [e]nd, [a]borted and reschedules, [s]uspended, [n]one
#$ -m n

# Validation level (e = reject on all problems)
#$ -w e

# Merge stdout and stderr streams: yes/no
#$ -j yes

# do not alter the task IDs below
#$ -t 1-30
#$ -tc 10
python thesisgenerator/plugins/experimental_utils.py $SGE_TASK_ID 
