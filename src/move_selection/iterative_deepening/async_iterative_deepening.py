import numpy as np
from time import time, sleep
from multiprocessing import Pipe
from multiprocessing.context import Process
from move_selection.move_chooser import MoveChooser
from move_selection.iterative_deepening.deepening_node import DeepeningNode


class AsyncIterativeDeepening(MoveChooser):
    def __init__(self, GameClass, starting_position=None, time_limit=3):
        super(AsyncIterativeDeepening, self).__init__(GameClass, starting_position)
        self.root = DeepeningNode(GameClass, starting_position)
        self.time_limit = time_limit

        self.parent_pipe, worker_pipe = Pipe()
        self.worker_process = Process(target=self.loop_func,
                                      args=(GameClass, starting_position, worker_pipe))
        self.receipt_backlog = 0

    def start(self):
        self.worker_process.start()

    def terminate(self):
        self.worker_process.terminate()
        self.worker_process.join()

    @staticmethod
    def loop_func(GameClass, starting_position, worker_pipe):
        root = DeepeningNode(GameClass, starting_position)
        while True:
            root.deepen()
            print(root.get_depth())
            worker_pipe.send(root.children[0].state)

            while worker_pipe.poll():
                chosen_position = worker_pipe.recv()
                if root.children is None:
                    root.deepen()

                for child in root.children:
                    if np.all(chosen_position == child.state):
                        root = child
                        break
                else:
                    raise ValueError('Invalid move!')
                worker_pipe.send(None)  # send acknowledgement receipt

    def report_user_move(self, user_chosen_position):
        # notify process that position was chosen
        self.parent_pipe.send(user_chosen_position)
        self.receipt_backlog += 1
        self.position = user_chosen_position

    def reset(self):
        raise NotImplementedError

    def choose_move(self, return_distribution=False):
        if return_distribution:
            raise NotImplementedError

        start_time = time()
        while self.receipt_backlog > 0:
            while self.parent_pipe.recv() is not None:
                pass
            self.receipt_backlog -= 1

        remaining_time = self.time_limit - (time() - start_time)
        if remaining_time > 0:
            sleep(remaining_time)

        self.position = self.parent_pipe.recv()
        while self.parent_pipe.poll():
            self.position = self.parent_pipe.recv()

        # notify process that position was chosen
        self.parent_pipe.send(self.position)
        self.receipt_backlog += 1

        return [self.position]
