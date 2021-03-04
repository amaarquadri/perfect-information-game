import os
import pickle
from time import time
from multiprocessing import Process, Pipe
import sys
from signal import signal, SIGTERM
import numpy as np
from heuristics.network import Network
from heuristics.proxy_network import ProxyNetwork
from utils.utils import get_training_path


def spawn_training_process(GameClass, model_path=None, threads=1):
    # double the number of threads so that half of them can work at a time, and then alternate
    parent_pipes, worker_pipes = zip(*[Pipe() for _ in range(2 * threads)])
    worker_training_data_pipe, parent_training_data_pipe = Pipe(duplex=False)
    process = Process(target=training_process_loop,
                      args=(GameClass, model_path, worker_pipes, worker_training_data_pipe))
    proxy_networks = [ProxyNetwork(GameClass, parent_a_pipe) for parent_a_pipe in parent_pipes]
    return process, proxy_networks, parent_training_data_pipe


def training_process_loop(GameClass, model_path, worker_pipes, training_data_pipe):
    network = Network(GameClass, model_path, reinforcement_training=True)
    network.initialize()

    def on_terminate_process(_):
        network.finish_training()
        network.save(model_path)
        sys.exit(0)
    signal(SIGTERM, on_terminate_process)

    last_save = 0  # initialize to 0 to ensure that the network is saved at the start
    while True:
        if training_data_pipe.poll():
            data = training_data_pipe.recv()
            states, policies, values = data
            network.train_step(states, policies, values)
            if time() - last_save > 30 * 60:  # every 30 minutes
                network.save(f'{model_path[:-3]}-{time()}.h5')
                last_save = time()

        # collect requests from whichever workers finish first
        request_indices = []
        requests = []
        while len(request_indices) < len(worker_pipes) // 2:
            for i, worker_pipe in enumerate(worker_pipes):
                if i not in request_indices and worker_pipe.poll():
                    request_indices.append(i)
                    requests.append(worker_pipe.recv())

        # concatenate is used instead of stack because each request already has shape (k,) + GameClass.STATE_SHAPE
        raw_policies, evaluations = network.predict(np.concatenate(requests, axis=0))

        # send the results back to the workers as fast as possible
        pos = 0
        for i, request in zip(request_indices, requests):
            new_pos = pos + request.shape[0]
            # This send call will not block because the receiver will always be waiting to read the result
            worker_pipes[i].send((raw_policies[pos:new_pos], evaluations[pos:new_pos]))
            pos = new_pos


def train_from_scratch(GameClass, hyper_params):
    net = Network(GameClass, hyper_params=hyper_params)
    net.initialize()
    print('Network size: ', net.model.count_params())

    def load_data(game_type):
        data = []
        for file in sorted(os.listdir(f'{get_training_path(GameClass)}/games/{game_type}')):
            if file[-7:] != '.pickle':
                continue
            with open(f'{get_training_path(GameClass)}/games/{game_type}/{file}', 'rb') as fin:
                data.append(pickle.load(fin))
        return data

    # net.train(load_data('rollout_mcts_games'))

    reinforcement_data = load_data('reinforcement_learning_games')[-750:]
    net.train(reinforcement_data)
    # sets = int(len(reinforcement_data) / 1200)
    # for k in range(sets):
    #     net.train(reinforcement_data[k * len(reinforcement_data) // sets:(k + 1) * len(reinforcement_data) // sets])

    net.save(f'{get_training_path(GameClass)}/models/model_reinforcement.h5')
