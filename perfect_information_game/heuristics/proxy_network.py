from perfect_information_game.heuristics import Network


class ProxyNetwork(Network):
    def __init__(self, GameClass, model_pipe):
        super().__init__(GameClass)
        self.model_pipe = model_pipe

    def initialize(self):
        pass

    def predict(self, states):
        self.model_pipe.send(states)
        return self.model_pipe.recv()

    def create_model(self, kernel_size=(4, 4), convolutional_filters=64, residual_layers=6,
                     value_head_neurons=16, policy_loss_value=1):
        raise NotImplementedError('ProxyNetwork does not support this operation!')

    def train(self, data, validation_fraction=0.2):
        raise NotImplementedError('ProxyNetwork does not support this operation!')

    def save(self, model_path):
        raise NotImplementedError('ProxyNetwork does not support this operation!')

    def equal_model_architecture(self, network):
        raise NotImplementedError('ProxyNetwork does not support this operation!')
