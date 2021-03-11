from games.chess import Chess


def get_verified_chess_subclass(GameClass):
    if GameClass is None:
        return Chess
    if issubclass(GameClass, Chess):
        return GameClass
    raise NotImplementedError('No support for classes that do not inherit from Chess!')
