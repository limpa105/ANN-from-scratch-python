from abc import ABC
from dataclasses import dataclass
from enum import Enum



    


@dataclass
class Neuron(ABC):
    """
    Represents fundamental Neuron Data Storage

    activation: float - default activation value
    layer: int  - the layer number which the neuron belongs to

    """
    layer: int
    activation: float = 0
    type: NeuronType = NeuronType.OUTPUT
    
    @staticmethod
    def get_type(self):
        return self.type

    def __str__(self):
        return self.__class__() + " "  + self.get_type()


@dataclass
class StandardNeuron(Neuron):
    """
    This is a standard neuron to be used in a simple neural network
    """
    pass
