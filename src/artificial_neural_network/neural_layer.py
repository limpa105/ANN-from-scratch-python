import numpy as np
from typing import Optional
from ..utils.utils import Utils
from neuron import StandardNeuron
from enum import Enum

class NeuralLayerType(Enum):
    """
    Enumerates Neural Layer Types
    """
    HIDDEN = "HIDDEN"
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


class NeuralLayer:
    def __init__(self, layer_num: int , layer_size: Optional[tuple] = (0,0), weights: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None):
        """
        Constructs a Neural Layer and randomly initializes weights and bias
        :param layer_num: an integer representing the layer number in the neural network
        :param layer_size: the (num_features, num_neurons) shape of the layer's weight matrix (for initialization)
        :param weights: optional pre-initialized layer weights
        :param bias: optional pre-initialized layer bias
        """
        self._layer_num = layer_num
        self._weights = weights
        self._bias = bias
        self.num_neurons, self.num_features = layer_size;

        # Setting Neural Layer type, will convert to hidden if new layer added
        if self._layer_num == 0:
            self.layer_type = NeuralLayerType.INPUT
        else:
            self.layer_type = NeuralLayerType.OUTPUT



        # Input validation
        if weights and bias:
            assert len(weights) == len(bias), "Weights and bias must have same number of features"

        if self.num_neurons and weights:
            assert len(weights[0]) == self.num_neurons, "Each weight needs to correspond to a neuron needs"

        if self.num_features and bias:
            assert len(bias) == self.num_features,  "Each bias needs to correspond to a feature"

        if self.num_features and weights:
            assert len(weights) == self.num_features, "Each weight needs to correspond to a feature"

        if not self.num_features and not weights:
            raise AttributeError("You must either pass a feature number or an initial weight matrix")

        # Initializing weights randomly - using num neurons passed / random
        if not weights:
            if not self.num_neurons:
                self.num_neurons = 5
            Utils.initialize_randomly(self._weights, [self.num_features, self.num_neurons])

        # Initializing bias with corresponding result from weight matrix
        if not bias:
            Utils.initialize_randomly(self._bias, [self.num_features])

        # Create number of new neurons and add them to the neural network
        for i in range(self.num_neurons):
            if layer_num == 0:
                self.add_neuron(StandardNeuron(weights[0][i]), 0, "INPUT")

            add_neuron(StandardNeuron())



     # Updatable weights
    @property
    def weights(self):
        return self._weights

    @property.setter
    def weights(self, new_weights):
        if not isinstance(new_weights,np.ndarray):
            raise AttributeError("Weights must be a valid matrix")
        self._weights = new_weights

    @property.getter
    def weights(self) -> np.ndarray:


    def add_neuron(neuron: StandardNeuron):
        pass

    def change_layer_state(self, new_state: NeuronType):