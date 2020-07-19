import numpy as np
from itertools import product


DIRECTIONS_8 = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def choose_random(values):
    return values[np.random.randint(len(values))]


def iter_product(shape, actions=None):
    values = tuple(range(k) for k in shape)
    if actions is not None:
        values += (actions,)
    return product(*values)


def test():
    for i, j, k, l in iter_product((2, 3, 4), ['a', 'b', 'c']):
        print(i, j, k, l)


if __name__ == '__main__':
    test()
