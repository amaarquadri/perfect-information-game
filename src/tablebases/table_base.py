import numpy as np
from games.chess import Chess, parse_fen, encode_fen


class Tablebase:
    DRAWING_TABLES = ['Kk', 'KBk', 'KNk']

    def __init__(self):
        self.tablebases = {}

    def query_position(self, state):
        if np.any(state[:, :, -2] == 1):
            return None

        # black is attacking, so switch white and black
        if Chess.heuristic(state) < 0:
            state = np.concatenate((state[:, :, 6:12], state[:, :, :6], -state[:, :, -2:]), axis=-1)

        fen = encode_fen(state)

    def notify_position(self, state):
        if np.all(state[:, :, -2] == 0) and sum(state[:, :, :12] != 0) <= 3:
            pass

    def load_table_base(self):
        pass
