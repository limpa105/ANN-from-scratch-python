from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Union
from src.artificial_neural_network.neural_layer import NeuralLayer
from ..utils.utils import Utils
from functools import lru_cache

class NeuralNetwork(ABC):

    @abstractmethod
    def train(self, epochs: int, batch_size: Optional[int], data:pd.DataFrame, labels:pd.Series):
        pass

    @abstractmethod
    def predict(self, data: Union[pd.DataFrame, np.ndarray]):
        pass


class ANN(NeuralNetwork):
    FEATURES_DIMENSION = 0

    def __init__(self, num_features: int, learning_rate: Optional[float] = 0.05, **hyperparameter_config):
        self.layers = []
        self.num_recent_features = num_features
        self.num_layers = 0
        self.learning_rate = learning_rate

    def train(self, epochs: int, data: pd.DataFrame, labels: pd.Series, batch_size: Optional[int] = None):
        if self.num_layers == 0:
            raise RuntimeError("Cannot train a model with no layers")
        assert np.unique(labels) == self.num_recent_features, """ The last layer's output dimensions need 
                                                                to match the number of classes"""
        unique_labels = np.unique(labels)
        self.label_enumeration = {unique_labels[i]: i for i in range(len(unique_labels))}
        self.reverse_enumerate = {value: key for key, value in self.label_enumeration.items()}

        for epoch in range(epochs):
            predictions = self._forward_propagate(data.to_numpy())
            cost = self._back_propagate(predictions, data.to_numpy(), labels.map(self.label_enumeration).to_numpy())
            print("cost", cost)

    def predict(self,  data: Union[np.ndarray, pd.DataFrame]):
        return self.reverse_enumerate[np.argmax(self._forward_propagate(data), axis=1)]

    def _forward_propagate(self, data: np.ndarray) -> np.ndarray:
        # first do the zero layer with
        activation = data * self.layers[0].weights + self.layers[0].bias
        activation = np.tanh(activation)
        self.layers[0].activations = activation
        for layer in self.layers[1:]:
            activation = np.tanh(activation * layer.weights + layer.bias)
            layer.activations = activation
        return activation

    def _error(self, predictions:np.ndarray, labels:np.ndarray):
        standarized = np.divide(predictions, np.sum(predictions, axis=1), axis=1)
        return Utils.cost(standarized, labels)

    @lru_cache(maxsize = 4)
    def _back_propagate(self, predictions: np.array, data: np.ndarray, labels: np.ndarray):
        # TODO: POTENTIAL MOVE TO lOGSPACE
        cost, cost_derivative = self._error(predictions, labels)
        final_layer = self.layers[-1]
        # Hadamard with the cost
        final_layer_derivative = cost_derivative * (1/np.arccosh(2 * final_layer.activations))
        final_layer.weights -= self.layers[-2].activations.T * final_layer_derivative
        final_layer.bias -= (np.ones(1, len(final_layer_derivative) * final_layer_derivative))

        def _back_propagate_helper(cached_derivative: float, layer: int):
            if layer != 0:
                next_derivative = ((cached_derivative *
                                    (self.layers[layer+1].weights.T))
                                    * (1/np.arccosh(2*self.layers[layer].activations)))

                self.layers[layer].weights -= self.layers[layer-1].activations.T * next_derivative
                self.layers[layer].bias -=  np.ones(1, len(final_layer_derivative)) * next_derivative
                return _back_propagate_helper(next_derivative, layer-1)
            else:
                return

        _back_propagate_helper(final_layer_derivative, len(self.layers)-1)
        return cost

    def add_layer(self, layer_size: Optional[tuple] = (0,0), weights: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None):
        if layer_size != (0, 0):
            if self.num_recent_features != layer_size[0]:
                raise AttributeError("layer feature lengths should match")
        elif weights is not None:
            if self.num_recent_features != len(weights):
                raise AttributeError("layer feature lengths should match")
        new_layer = NeuralLayer(self.num_layers, layer_size, weights, bias)
        self.num_layers += 1
        np.add(self.layers, new_layer)
        self.num_recent_features = len(new_layer.weights[ANN.FEATURES_DIMENSION])


