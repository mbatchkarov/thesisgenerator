# coding=utf-8
__author__ = 'mmb28'

# with open('/Volumes/LocalScratchHD/LocalHome/NetBeansProjects/thesisgenerator'
#           '/conf/exp5/logs/exp5-11.log') as infile:
#     txt = ''.join(infile.readlines())
#
# import re
# unknowns = set(re.findall(r'Unknown token in doc \d+: (.*/.*)',txt))
# founds = set(re.findall(r'Found thesaurus entry for (.*/.*)',txt))
# print len(unknowns)
# print len(founds)
#
# print '\n'.join(unknowns-founds)




# import random
# def may_fail():
#     if random.random() < 0.8:
#         raise ValueError('Hahaha')
#     else:
#         print 'success'
#         return 0
#
#
# from joblib import Parallel, delayed
#
# try:
#     Parallel(n_jobs=2)(delayed(may_fail)() for _ in range(100))
# except joblib.my_exceptions.JoblibValueError:
#     print 'failed, but handled'

def prebyblo_filter(input_file, filtered_tokens):
    with open(filtered_tokens) as fh:
        filtered = set(token.strip() for token in fh)

    with open(input_file) as infile:
        with open('%s.filtered' % input_file, 'w') as outfile:
            for line in infile:
                if line.split('\t')[0] not in filtered:
                    outfile.write(line)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print 'Usage: filter.py input_file tokens_to_filter'
        sys.exit(1)

    input_file = sys.argv[1]
    filtered_tokens = sys.argv[2]
    prebyblo_filter(input_file, filtered_tokens)