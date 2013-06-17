from joblib import Memory
from thesisgenerator.utils import NoopTransformer


def init_cache(enabled):
    global memory
    if enabled:
        memory = Memory(cachedir='.', verbose=0)
    else:
        memory = NoopTransformer()
    return memory


def get_memory():
    return memory