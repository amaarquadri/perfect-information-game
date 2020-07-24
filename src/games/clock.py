from time import time
from multiprocessing import Process, Pipe, Event


class Clock:
    def __init__(self, starting_time=5 * 60, increment=5):
        self.player_1_event = Event()
        self.player_2_event = Event()
        self.timing_update_pipe, worker_timing_update_pipe = Pipe(False)
        self.timing_process = Process(target=Clock.timing_process_loop, args=(starting_time, increment,
                                                                              self.player_1_event, self.player_2_event,
                                                                              worker_timing_update_pipe))

    def start(self):
        self.timing_process.start()

    def terminate(self):
        self.timing_process.terminate()

    def player_1_move(self):
        self.player_1_event.set()
        while self.player_1_event.is_set():
            pass

    def player_2_move(self):
        self.player_2_event.set()
        while self.player_2_event.is_set():
            pass

    @staticmethod
    def timing_process_loop(starting_time, increment, player_1_event, player_2_event, timing_update_pipe):
        player_1_time = starting_time
        player_2_time = starting_time
        while True:


            move_start_time = time()
            if player_1_event.wait(player_1_time):
                player_1_time += increment - (time() - move_start_time)
                player_1_event.clear()
            else:
                # player 1 timed out
                break

            move_start_time = time()
            if player_2_event.wait(player_1_time):
                player_2_time += increment - (time() - move_start_time)
                player_2_event.clear()
            else:
                # player 2 timed out
                break
