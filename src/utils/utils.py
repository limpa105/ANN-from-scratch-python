from numpy import ndarray
from numpy import random
from numpy import ndarray


class Utils:
    @staticmethod
    def initialize_randomly(self, shape: list) -> ndarray:
        """

        :param shape: a list representing shape to initialize randomly
        :return: a randomly initialized ndarray
        """
        return random.rand(*shape)

    @staticmethod
    def cost(self, predictions: ndarray, labels: ndarray):
        errors = [1-predictions[label] for label in labels]
        return sum(errors)

    



class Matrix_Utils:
    pass
