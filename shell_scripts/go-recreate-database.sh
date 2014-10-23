#!/bin/sh
python thesisgenerator/utils/db.py # drop and re-create all tables
python thesisgenerator/scripts/populate_vectors_db.py # find vector files
python thesisgenerator/scripts/generate_classification_conf_files.py # insert experiment entries in DB
python thesisgenerator/plugins/dumpers.py # scrape results from disk and store in DB