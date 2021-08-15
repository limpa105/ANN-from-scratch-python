import numpy as np
from typing import Optional, Union
from ..utils.utils import Utils
from enum import Enum



class NeuralLayerType(Enum):
    """
    Enumerates Neural Layer Types
    """
    HIDDEN = "HIDDEN"
    OUTPUT = "OUTPUT"


class NeuralLayer:
    def __init__(self, layer_num: int, layer_size: Optional[tuple] = (0,0), weights: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None):
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
        self.num_neurons, self.num_features = layer_size
        self._activations = None

        # Setting Neural Layer type, will convert to hidden if new layer added
        self._layer_type = NeuralLayerType.OUTPUT

        # Input validation
        if weights is not None and bias is not None:
            assert len(weights) == len(bias), "Weights and bias must have same number of features"

        if self.num_neurons and weights is not None:
            assert len(weights[0]) == self.num_neurons, "Each weight needs to correspond to a neuron needs"

        if self.num_neurons and bias is not None:
            assert len(bias) == self.num_features,  "Each bias needs to correspond to a neuron"

        if self.num_features and weights is not None:
            assert len(weights) == self.num_features, "Each weight needs to correspond to a feature"

        if not self.num_features and weights is None:
            raise AttributeError("You must either pass a feature number or an initial weight matrix")

        # Initializing weights randomly - using num neurons passed / random
        if weights is None:
            if not self.num_neurons:
                self.num_neurons = 5
            self._weights = Utils.initialize_randomly(np.ndarray(self.num_features, self.num_neurons))

        # Initializing bias with corresponding result from weight matrix
        if bias is None:
            if not self.num_neurons and weights is not None:
                self.num_neurons = len(weights[0])
            self._bias = Utils.initialize_randomly(np.ndarray(self.num_neurons))

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        if not isinstance(new_weights,np.ndarray):
            raise AttributeError("Weights must be a valid matrix")
        self._weights = new_weights

    @weights.getter
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        if not isinstance(new_bias,np.ndarray):
            raise AttributeError("Bias must be a valid matrix")
        self._bias = new_bias

    @bias.getter
    def bias(self) -> Union[np.ndarray, np.array]:
        return self._bias

    @property
    def layer_type(self):
        return self._layer_type

    @layer_type.setter
    def layer_type(self, new_type):
        if not isinstance(new_type, NeuralLayerType):
            raise AttributeError("Neural Layer Type must be a valid NeuralLayerType")
        self._layer_type = new_type

    @layer_type.getter
    def layer_type(self) -> NeuralLayerType:
        return self._layer_type

    @property
    def activations(self):
        return self._activations

    @activations.setter
    def activations(self, activations_input: np.ndarray):
        if self._activations is not None and not isinstance(activations_input, np.ndarray):
            raise AttributeError("Activation must be a matrix")
        self._activations = activations_input

    @activations.getter
    def activations(self) -> np.ndarray:
        return self._activations
    

