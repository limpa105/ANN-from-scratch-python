from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Union
from src.artificial_neural_network.neural_layer import NeuralLayer
from ..utils.utils import Utils
from functools import lru_cache
import math

class NeuralNetwork(ABC):

    @abstractmethod
    def train(self, epochs: int, data:pd.DataFrame, labels:pd.Series):
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

    def train(self, epochs: int, data: pd.DataFrame, labels: pd.Series):
        """
        Given input data, trains neural network model and prints the cost

        :param epochs: number of epochs the model should be tested for
        :param data: training data
        :param labels: expected labels corresponding to training records
        """
        if self.num_layers == 0:
            raise RuntimeError("Cannot train a model with no layers")
        assert len(np.unique(labels)) == self.num_recent_features, """ The last layer's output dimensions need 
                                                                to match the number of classes"""
        unique_labels = np.unique(labels)
        self.label_enumeration = {unique_labels[i]: i for i in range(len(unique_labels))}
        self.reverse_enumerate = {value: key for key, value in self.label_enumeration.items()}
        for col in data.columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()

        for epoch in range(epochs):
            predictions = self._forward_propagate(data.to_numpy())
            cost = self._back_propagate(predictions, labels.map(self.label_enumeration).to_numpy())
            print("cost", cost)

    def predict(self,  data: Union[np.ndarray, pd.DataFrame]):
        """
        Returns a set of predictions given a set of inputs
        :param data: A set of featured examples to make predictions on
        :return: A set of predictions
        """
        data = pd.DataFrame(data)
        for col in data.columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
        data = data.to_numpy()
        predictions = np.argmax(self._forward_propagate(data), axis=1)
        return [self.reverse_enumerate[prediction] for prediction in predictions]

    def _forward_propagate(self, data: np.ndarray) -> np.ndarray:
        activation = np.matmul(data,self.layers[0].weights) + self.layers[0].bias
        activation = np.tanh(activation)
        self.layers[0].activations = activation
        for layer in self.layers[1:]:
            if layer is self.layers[-1]:
                activation = self._logistic_sigmoid(np.matmul(activation,layer.weights) + layer.bias)
            else:
                activation = np.tanh(np.matmul(activation, layer.weights) + layer.bias)
            layer.activations = activation
        return activation

    def _logistic_sigmoid(self, activation:np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-1 * activation))


    def _error(self, predictions:np.ndarray, labels:np.ndarray):
        return Utils.cost(predictions, labels)

    def _back_propagate(self, predictions: np.ndarray, labels: np.ndarray):
        cost, cost_derivative = self._error(predictions, labels)
        final_layer = self.layers[-1]
        activation_der = self._logistic_sigmoid(final_layer.activations)*(1 - self._logistic_sigmoid(final_layer.activations))
        final_layer_derivative = cost_derivative * activation_der
        final_layer.weights -= np.matmul(self.layers[-2].activations.T,final_layer_derivative)
        final_layer.bias -= np.matmul(np.ones(shape=(1, len(final_layer_derivative))), final_layer_derivative)

        def _back_propagate_helper(cached_derivative: float, layer: int):
            if layer != 0:
                act_der = (1 - (np.tanh(self.layers[layer].activations) ** 2))
                next_derivative = ((np.matmul(cached_derivative,
                                    self.layers[layer+1].weights.T) *
                                    act_der))

                self.layers[layer].weights -= np.matmul(self.layers[layer-1].activations.T,next_derivative)
                self.layers[layer].bias -= np.matmul(np.ones(shape=(1, len(final_layer_derivative))),next_derivative)
                return _back_propagate_helper(next_derivative, layer-1)
            else:
                return

        _back_propagate_helper(final_layer_derivative, len(self.layers)-2)
        return cost

    def add_layer(self, layer_size: Optional[tuple] = (0,0), weights: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None):
        """
        Adds a layer to the neural network

        :param layer_size: A tuple representing the input dimensions of the layer
        :param weights: A set of initial / pretrained weights, optional
        :param bias: An initial /pretrained bias
        """
        if layer_size != (0, 0):
            if self.num_recent_features != layer_size[0]:
                raise AttributeError("layer feature lengths should match")
        elif weights is not None:
            if self.num_recent_features != len(weights):
                raise AttributeError("layer feature lengths should match")
        new_layer = NeuralLayer(self.num_layers, layer_size, weights, bias)
        self.num_layers += 1
        self.layers = np.append(self.layers, new_layer)

        self.num_recent_features = len(new_layer.weights[0])


