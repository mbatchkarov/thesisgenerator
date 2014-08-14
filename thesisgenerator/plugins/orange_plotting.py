from itertools import combinations
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
# code modified from Orange:
# https://bitbucket.org/biolab/orange/src/a4303110189426d004156ce053ddb35a410e428a/Orange/evaluation/scoring.py

def nth(l, n):
    """
    Returns only nth elemnt in a list.
    """
    n = lloc(l, n)
    return [a[n] for a in l]


def lloc(l, n):
    """
    List location in list of list structure.
    Enable the use of negative locations:
    -1 is the last element, -2 second last...
    """
    if n < 0:
        return len(l[0]) + n
    else:
        return n


def print_figure(fig, *args, **kwargs):
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(*args, **kwargs)


def get_lines(scores, cd):
    """
    Get pairs of non significant methods

    :param scores: a list of float scores for each method
     :type scores: list
    :param cd: how large the difference between a pair of methods has to be to be significant
    :return: list of tuples of indices of all pairs of methods that are not significantly different and
    should be connected in the diagram. Each tuple must be sorted, and no duplicate tuples should be contained.

    Examples:
     - [(0, 1), (3, 4), (4, 5)] is correct
     - [(0, 1), (3, 4), (4, 5), (3,4)] contains a duplicate
     - [(3, 4)] contains a non-sorted tuple

    TODO if there is a cluster of non-significant differences, return as a tuple the idx of the smallest and largest
     values in the cluster only
    """

    # get all pairs
    n_methods = len(scores)
    allpairs = list(combinations(range(n_methods), 2))

    # todo mmb- this is the crucial bit that determines what is connected and what is not
    # remove not significant
    not_sig = [(i, j) for i, j in allpairs if abs(scores[i] - scores[j]) <= cd]

    # keep only longest
    def no_longer(i, j, notSig):
        for i1, j1 in notSig:
            if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                return False
        return True

    longest = [(i, j) for i, j in not_sig if no_longer(i, j, not_sig)]

    return longest


def graph_ranks(avranks, names, cd=None, cdmethod=None, lowv=None, highv=None, width=6, textspace=1.5,
                reverse=False, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods' 
    performance.
    See Janez Demsar, Statistical Comparisons of Classifiers over 
    Multiple Data Sets, 7(Jan):1--30, 2006. 

    Needs matplotlib to work.

    :param filename: Output file name (with extension). Formats supported 
                     by matplotlib can be used.
    :param avranks: List of average methods' ranks.
    :param names: List of methods' names.

    :param cd: Critical difference. Used for marking methods that whose
               difference is not statistically significant.
    :param lowv: The lowest shown rank, if None, use 1.
    :param highv: The highest shown rank, if None, use len(avranks).
    :param width: Width of the drawn figure in inches, default 6 in.
    :param textspace: Space on figure sides left for the description
                      of methods, default 1 in.
    :param reverse:  If True, the lowest rank is on the right. Default\: False.
    :param cdmethod: None by default. It can be an index of element in avranks
                     or or names which specifies the method which should be
                     marked with an interval.
    """
    width = float(width)
    textspace = float(textspace)

    sums = avranks

    tempsort = sorted([(a, i) for i, a in enumerate(sums)], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [names[x] for x in sortidx]

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    if cd and cdmethod is None:
        # get pairs of non significant methods
        lines = get_lines(ssums, cd)
        linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

        # add scale
        distanceh = 0.25
        cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = Figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]


    # Upper left corner is (0,0).

    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    # main axis
    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    import numpy

    tick = None
    # ticks on the x-axis
    for a in list(numpy.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a): tick = bigtick
        line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=0.7)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a), ha="center", va="bottom")

    k = len(ssums)

    # 'arrows' pointing to the first half of the method names
    for i in range((k + 1) / 2):
        chei = cline + minnotsignificant + i * 0.2
        line([(rankpos(ssums[i]), cline), (rankpos(ssums[i]), chei), (textspace - 0.1, chei)], linewidth=0.7)
        text(textspace - 0.2, chei, nnames[i], ha="right", va="center")
    # arrows poiting to the second half of method names
    for i in range((k + 1) / 2, k):
        chei = cline + minnotsignificant + (k - i - 1) * 0.2
        line([(rankpos(ssums[i]), cline), (rankpos(ssums[i]), chei), (textspace + scalewidth + 0.1, chei)],
             linewidth=0.7)
        text(textspace + scalewidth + 0.2, chei, nnames[i], ha="left", va="center")

    if cd and cdmethod is None:
        # if we want to annotate a single method with the critical difference

        # upper scale
        if not reverse:
            begin, end = rankpos(lowv), rankpos(lowv + cd)
        else:
            begin, end = rankpos(highv), rankpos(highv - cd)

        # draw a line as large as the CD above the main plot to give a sense of scale
        line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
        line([(begin, distanceh + bigtick / 2), (begin, distanceh - bigtick / 2)], linewidth=0.7)
        line([(end, distanceh + bigtick / 2), (end, distanceh - bigtick / 2)], linewidth=0.7)
        text((begin + end) / 2, distanceh - 0.05, "CD", ha="center", va="bottom")

        def draw_lines(lines, side=0.05, height=0.1):
            start = cline + 0.2
            for l, r in lines:
                line([(rankpos(ssums[l]) - side, start), (rankpos(ssums[r]) + side, start)], linewidth=2.5)
                start += height

        # draw the lines that connect methods that are not significantly different
        draw_lines(lines)

    elif cd:
        # draw a single fat line on the x axis centered around `cdmethod`
        begin = rankpos(avranks[cdmethod] - cd)
        end = rankpos(avranks[cdmethod] + cd)
        line([(begin, cline), (end, cline)], linewidth=2.5)
        line([(begin, cline + bigtick / 2), (begin, cline - bigtick / 2)], linewidth=2.5)
        line([(end, cline + bigtick / 2), (end, cline - bigtick / 2)], linewidth=2.5)

    return fig


def my_get_lines(*args, **kwargs):
    return [(3, 4), (3, 5)]


if __name__ == "__main__":
    avranks = [3.143, 2.000, 2.893, 1.964, 2.5, 3.34]
    names = ["prva", "druga", "tretja", "cetrta", 'peta', 'shesta']
    cd = 0.3

    get_lines = my_get_lines
    fig = graph_ranks(avranks, names, cd=cd)  # ha! cd needs to be small to see it
    # fig = graph_ranks("test.eps", avranks, names, cd=cd, cdmethod=4)

    print_figure(fig, "test.eps")
