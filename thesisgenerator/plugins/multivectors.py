from functools import lru_cache
import numpy as np
import pandas as pd
from discoutils.thesaurus_loader import Vectors


class MultiVectors(Vectors):
    def __init__(self, vectors):
        self.vectors = vectors

    def init_sims(self, *args, **kwargs):
        for v in self.vectors:
            v.init_sims(*args, **kwargs)

    def __len__(self):
        return len(self.vectors[0])

    def __contains__(self, item):
        return any(item in v for v in self.vectors)

    @lru_cache(maxsize=2 ** 16)
    def get_nearest_neighbours(self, entry):
        if entry not in self:
            return []

        if sum(entry in v for v in self.vectors) < 2:
            # entry contained in too few of the repeated runs, it is probably a spurious word
            # with a low-quality vector. pretend it is not there
            return []
        data = []
        for tid, t in enumerate(self.vectors):
            neighbours = t.get_nearest_neighbours_linear(entry)
            if neighbours:
                for rank, (neigh, sim) in enumerate(neighbours):
                    data.append([tid, rank, neigh, sim])
        if not data:
            return []
        df = pd.DataFrame(data, columns='tid, rank, neigh, sim'.split(', '))

        # Watch out! Higher rank is currently better! This makes sense if we use this is a similarity metric or a
        # pseudo term count, but doesn't match the rest of the codebase, where distances are used (lower is better)
        ddf = df.groupby('neigh').aggregate({'rank': 'mean',
                                             'sim': 'mean',
                                             'tid': 'count'}).rename(columns={'tid': 'contained_in',
                                                                              'rank': 'mean_rank',
                                                                              'sim': 'mean_dist'})
        ddf = ddf.sort(['contained_in', 'mean_rank', 'mean_dist'],
                       ascending=[False, True, True], kind='mergesort')  # must be stable
        a = 1. + ddf.mean_rank.values
        ddf.mean_rank = 1 / a
        return list(zip(ddf.index, ddf.mean_rank) if len(ddf) else [])
