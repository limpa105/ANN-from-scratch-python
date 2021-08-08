from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Union
from neural_layer import NeuralLayer


class NeuralNetwork(ABC):

    @abstractmethod
    def train(self, epochs: int, batch_size: Optional[int], data:pd.DataFrame, labels:pd.Series):
        pass

    @abstractmethod
    def predict(self, data: Union[pd.DataFrame, np.ndarray]):
        pass

class ANN(NeuralNetwork):
    FEATURES = 0
    def __init__(self, num_features: int, learning_rate: Optional[int] = 0.05, **hyperparameter_config):
        self.layers = np.array([])
        self.num_recent_features = num_features
        self.num_layers = 0
        self.learning_rate = learning_rate

    def train(self, epochs: int, batch_size: Optional[int], data:pd.DataFrame, labels: pd.Series):
        #
        if self.num_layers == 0:
            raise RuntimeError("Cannot train a model with no layers")

        for epoch in range(epochs):
            predictions = self._forward_propagate(data.to_numpy())
            self._back_propagate(predictions, labels.to_numpy())

    def predict(self,  data: Union[np.ndarray, pd.DataFrame]):
        return np.argmax(self._forward_propagate(data), axis=1)

    def _forward_propagate(self, data: np.ndarray) -> np.ndarray:
        # first do the zero layer with
        activation = data * self.layers[0].weights + self.layers[0].bias
        activation = np.tanh(activation)
        for layer in self.layers[1:]:
            activation = np.tanh(activation * layer.weights + layer.bias)
        return activation
    
    def _error(self, predictions:np.ndarray, labels:np.ndarray):


    def _back_propagate(self, predictions: np.ndarray, labels: np.ndarray):
        pass

    def add_layer(self, layer_size: Optional[tuple] = (0,0), weights: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None):
        if layer_size:
            if self.num_recent_features != layer_size[0]:
                raise AttributeError("layer feature lengths should match")
        elif weights:
            if self.num_recent_features != len(weights):
                raise AttributeError("layer feature lengths should match")
        new_layer = NeuralLayer(self.num_layers, layer_size, weights, bias)
        self.num_layers += 1
        np.add(self.layers, new_layer)
        self.num_recent_features = new_layer.weights[ANN.FEATURES]


