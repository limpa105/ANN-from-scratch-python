from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Union
from neural_layer import NeuralLayer
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
    FEATURES = 0

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

    def predict(self,  data: Union[np.ndarray, pd.DataFsrame]):
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

    @lru_cache(maxsize=1000)
    def _back_propagate(self, predictions: pd.np.array, data: np.ndarray, labels: np.ndarray):
        cost = self._error(predictions, labels)
        recent_derivative = 2 * np.sum([(1-predictions[label]) for label in labels])
        reversed_layers = self.layers[::-1]
        for i in range(1, len(reversed_layers)-1):
            current_layer = reversed_layers[i]
            previous_layer = reversed_layers[i-1]
            current_layer.weights = current_layer.weights - self.learning_rate * \
                                    np.sum(recent_derivative *
                                     previous_layer.activations *
                                     (1/np.arccosh(2 * current_layer.activations)), axis = 1)

            current_layer.bias = (current_layer.bias - self.learning_rate * np.sum(
                                  recent_derivative *
                                  (1/np.arccosh(2 * current_layer.activations)), axis = 1))

            recent_derivative = np.sum((recent_derivative * current_layer.weights * (1/np.arccosh(2 * current_layer.activations))), axis = 1)

        first_layer = self.layers[0]
        means = [col.mean() for col in data.columns()]
        first_layer.weights = recent_derivative * (1/np.arccosh(2 * first_layer.activations)) * means
        return cost

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


