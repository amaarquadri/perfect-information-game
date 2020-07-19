from time import sleep


class RawNetwork:
    def __init__(self, network=None, delay=2):
        self.network = network
        self.delay = delay

    def start(self):
        self.network.initialize()

    def choose_move(self, user_chosen_position):
        sleep(self.delay)
        return self.network.choose_move(user_chosen_position)

    def terminate(self):
        pass
