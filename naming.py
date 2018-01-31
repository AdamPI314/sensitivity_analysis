"""
sensitivity analysis naming
"""


def index_transition(n_dim, exclude=None):
    """
    suppose orginal list is [0, 1, ..., n_dim - 1]
    exclude = [0, 2]
    return two dictionaries,
    1), original to new dictionary
    2), and new to original dictionary
    """
    o_2_n = dict()
    n_2_o = dict()

    counter = 0
    for i in range(n_dim):
        if i not in exclude:
            o_2_n[i] = counter
            n_2_o[counter] = i
            counter += 1
        else:
            continue
    return o_2_n, n_2_o
