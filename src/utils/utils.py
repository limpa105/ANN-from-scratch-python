from numpy import ndarray
from numpy import random
from numpy import ndarray
from numpy import mean as np_mean


class Utils:
    @staticmethod
    def initialize_randomly(shape: list) -> ndarray:
        """

        :param shape: a list representing shape to initialize randomly
        :return: a randomly initialized ndarray
        """
        return random.rand(*shape)

    @staticmethod
    def cost(predictions: ndarray, labels: ndarray):
        errors = [(1-predictions[label])**2 for label in labels]
        return np_mean(errors)

    



class Matrix_Utils:
    pass
