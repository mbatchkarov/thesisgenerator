#!/bin/bash

# Parameters for Sun Grid Engine submition
# ============================================

# Shell to use
#$ -S /bin/bash

# All paths relative to current working directory
#$ -cwd

# List of queues- the last two are high memory and not really suitable for this job
#$ -q nlp-amd,eng-inf_parallel.q,parallel.q,serial.q,inf.q,eng-inf_himem.q

# Define parallel environment for N cores
#$ -pe openmp 1

# Send mail to. (Comma separated list)
#$ -M mmb28@sussex.ac.uk

# When: [b]eginning, [e]nd, [a]borted and reschedules, [s]uspended, [n]one
#$ -m n

# Validation level (e = reject on all problems)
#$ -w e

# Merge stdout and stderr streams: yes/no
#$ -j yes

cd /lustre/scratch/inf/mmb28/DiscoUtils
python discoutils/prebyblo_filter.py -pos N V J $@
