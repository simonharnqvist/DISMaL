import itertools

def tally_to_counts(x):
    return list(itertools.chain(*[[i] * x[i] for i in range(0, len(x))]))