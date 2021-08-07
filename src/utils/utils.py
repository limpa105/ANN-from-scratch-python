from numpy import ndarray
from numpy import random


class Utils:
    @staticmethod
    def initialize_randomly(self, shape: list) -> ndarray:
        """

        :param shape: a list representing shape to initialize randomly
        :return: a randomly initialized ndarray
        """
        return random.rand(*shape)


class Matrix_Utils:
    pass