import numpy as np
from itertools import product
from multiprocessing import Pool


class OptionalPool:
    def __init__(self, threads=1):
        self.pool = Pool(threads) if threads > 1 else None

    def map(self, func, iterable, chunksize=None):
        return self.pool.map(func, iterable, chunksize) if self.pool is not None else map(func, iterable)

    def imap_unordered(self, func, iterable, chunksize=1):
        """
        If self.pool is not None, then:
        items are taken from the iterable lazily and
        the result is not guaranteed to be in the corresponding order of the iterable
        """
        return self.pool.imap_unordered(func, iterable, chunksize) if self.pool is not None else map(func, iterable)

    def starmap(self, func, iterable, chunksize=None):
        return self.pool.starmap(func, iterable, chunksize) if self.pool is not None \
            else [func(*params) for params in iterable]

    def __enter__(self):
        if self.pool is not None:
            self.pool.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool is not None:
            self.pool.__exit__(exc_type, exc_val, exc_tb)
            
    def close(self):
        if self.pool is not None:
            self.pool.close()

    def join(self):
        if self.pool is not None:
            self.pool.join()


STRAIGHT_DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRECTIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
DIRECTIONS_8 = STRAIGHT_DIRECTIONS + DIAGONAL_DIRECTIONS


def get_training_path(GameClass):
    path = f'../../training/{GameClass.__name__}'
    ruleset = GameClass.get_ruleset()
    return path if ruleset is None else f'{path}/{ruleset}'


def choose_random(values):
    return values[np.random.randint(len(values))]


def one_hot(index, size, dtype=None):
    result = np.zeros(size, dtype=dtype)
    result[index] = 1
    return result


def iter_product(shape, actions=None):
    values = tuple(range(k) for k in shape)
    if actions is not None:
        values += (actions,)
    return product(*values)


def get_np_uint_type(bits):
    if bits == 8:
        return np.uint8
    if bits == 16:
        return np.uint16
    if bits == 32:
        return np.uint32
    if bits == 64:
        return np.uint64
    raise ValueError(f'No numpy data type for {bits} bit unsigned integer!')


def alternate_iterables(*iterables):
    """
    This function is a generator that alternates between yielding elements from the given iterables
    """
    iterators = [iter(iterable) for iterable in iterables]

    while True:
        try:
            for iterator in iterators:
                yield next(iterator)
        except StopIteration:
            break


def test():
    for i, j, k, l in iter_product((2, 3, 4), ['a', 'b', 'c']):
        print(i, j, k, l)


if __name__ == '__main__':
    test()
