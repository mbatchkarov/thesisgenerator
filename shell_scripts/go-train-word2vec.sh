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
#$ -pe openmp 7

# Send mail to. (Comma separated list)
#$ -M mmb28@sussex.ac.uk

# When: [b]eginning, [e]nd, [a]borted and reschedules, [s]uspended, [n]one
#$ -m eas

# Validation level (e = reject on all problems)
#$ -w e

# Merge stdout and stderr streams: yes/no
#$ -j yes

#$ -N word2vec

#$ -t 1-13
#$ -tc 15

SETTINGS_FILE=mmb28_w2v_settings.tmp.$RANDOM # python job below is invoked for each task in the jobs array, write to a separate file to avoid conflicts
python -c "
for i in [1, 15] + list(range(10, 101, 10)):
    if i in [15, 50]:
        print('--corpus wiki --stages vectors compose average --percent %d --repeat 3'%i)
    else:
        print('--corpus wiki --stages vectors compose --percent %d'%i)
    
print('--corpus gigaw --stages vectors compose average --percent 100 --repeat 3')
" > $SETTINGS_FILE

SEED=$(awk "NR==$SGE_TASK_ID" $SETTINGS_FILE) 
echo $SEED
python thesisgenerator/scripts/get_word2vec_vectors.py $SEED
