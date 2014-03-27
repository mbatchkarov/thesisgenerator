from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd



def histogram_from_list(l, path):
    MAX_LABEL_COUNT = 20

    plt.figure()
    if type(l[0]) == str:
        s = pd.Series(Counter(l))
        s.plot(kind='bar', rot=0)
    else:
        plt.hist(l, bins=MAX_LABEL_COUNT)

    plt.savefig(path, format='png')
