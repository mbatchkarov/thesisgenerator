import argparse
import logging
import iterpipes


def set_stage_in_byblo_conf_file(filename, stage_id):
    """
    Add/removes the --stages switch from a Byblo conf filename
    :param stage_id: 0 if the --stages information should be removed, 1 if it has to be set to the first stage of
     Byblo (vector creation) and 2 for the second stage (all-pairs similarity)
    """
    with open(filename) as inf:
        lines = [x.strip() for x in inf.readlines()]
    stages = {
        0: '', # run the entire Byblo pipeline
        1: ['--stages', 'enumerate,count,filter'], # run the first part only
        2: ['--stages', 'allpairs,knn,unenumerate'] # run the second part only
    }

    # remove the current stages setting, may be multiple
    while True:
        try:
            index = lines.index('--stages')
            lines.pop(index)
            lines.pop(index)
        except ValueError:
            # '--stages' is not in list, nothing more to do
            break

    with open(filename, "w") as outf:
        for line in lines:
            outf.write(line)
            outf.write('\n')
        for line in stages[stage_id]:
            outf.write(line)
            outf.write('\n')


def set_output_in_byblo_conf_file(filename, new_output_prefix):
    with open(filename) as inf:
        lines = [x.strip() for x in inf.readlines()]

    try:
        index = lines.index('--output')
    except ValueError:
        # not there, try the short form
        try:
            index = lines.index('-o')
        except ValueError:
            raise ValueError('Cannot find the "output" parameter to Byblo in file %s' % filename)
    lines.pop(index)
    lines.pop(index)

    with open(filename, "w") as outf:
        outf.write('--output\n')
        outf.write('%s\n' % new_output_prefix)

        for line in lines:
            outf.write(line)
            outf.write('\n')


def parse_byblo_conf_file(path):
    """
    Parses a byblo conf file (switch per line) and extracts a few important switches. Has only been configured
    to recognise

    :returns: a tuple of (known, unknown) arguments
    """
    with open(path) as infile:
        lines = ' '.join([x.strip() for x in infile.readlines()])
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", '--output', type=str)
    parser.add_argument("-i", '--input', type=str)
    parser.add_argument("-fef", '--filter-entry-freq', type=int)
    parser.add_argument("-fff", '--filter-feature-freq', type=int)
    parser.add_argument('--stages', type=str)
    args = parser.parse_known_args(lines.split(' '))
    return args


def run_and_log_output(cmd_string):
    """
    Runs a command with iterpipes and logs the output
    """
    logging.info('Running %s', cmd_string)
    c = iterpipes.cmd(cmd_string)
    out = iterpipes.run(c)
    for line in out:
        logging.info(line)


def run_byblo(conf_file):
    run_and_log_output('./byblo.sh @{}'.format(conf_file))


def unindex_all_byblo_vectors(outfile_name):
    """
    unindexes byblo's vector files to a string representation

    :param outfile_name: the name of the output file used when these vector files were produced
    """
    run_and_log_output(
        './tools.sh unindex-events -i {0}.events.filtered -o {0}.events.filtered.strings '
        '-Xe {0}.entry-index -Xf {0}.feature-index -et JDBM'.format(outfile_name))
    run_and_log_output(
        './tools.sh unindex-features -et JDBM  -i {0}.features.filtered  '
        '-o {0}.features.filtered.strings  -Xf {0}.feature-index -Ef'.format(outfile_name))
    run_and_log_output(
        './tools.sh unindex-entries -et JDBM  -i {0}.entries.filtered  '
        '-o {0}.entries.filtered.strings  -Xe {0}.entry-index -Ee'.format(outfile_name))

    # remove the __FILTERED__ feature, entry and event so that it doesn't mess with cosine similarity
    for file_type in ['entries', 'features']:
        my_file = '{}.{}.filtered.strings'.format(outfile_name, file_type)
        with open(my_file) as infile:
            lines = infile.readlines()

        with open(my_file, 'w+b') as outfile:
            for line in lines:
                if '__FILTERED__' not in line:
                    outfile.write(line)
                else:
                    logging.info('Removed line %s from %s', line.strip(), my_file)

    events_file = '{}.events.filtered.strings'.format(outfile_name)
    with open(events_file) as infile:
        lines = infile.readlines()

    with open(events_file, 'w+b') as outfile:
        for line in lines:
            if not line.startswith('___FILTERED___'):
                outfile.write('\t'.join(line.split('\t')[:-2]))
                outfile.write('\n')
            else:
                logging.info('Removed line %s from %s', line.strip(), events_file)


def reindex_all_byblo_vectors(output_prefix):
    """rebuild index from a string representation"""
    run_and_log_output('./tools.sh index-features -et JDBM  -i {0}.features.filtered.strings  '
                       '-o {0}.features.filtered -Xf {0}.feature-index'.format(output_prefix))
    run_and_log_output('./tools.sh index-entries -et JDBM  -i {0}.entries.filtered.strings '
                       '-o {0}.entries.filtered -Xe {0}.entry-index'.format(output_prefix))
    run_and_log_output('./tools.sh index-events -et JDBM -i {0}.events.filtered.strings '
                       '-o {0}.events.filtered -Xe {0}.entry-index -Xf {0}.feature-index'.format(output_prefix))
