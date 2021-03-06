from numpy import ndarray
from numpy import random
from numpy import ndarray
import numpy as np


class Utils:
    @staticmethod
    def initialize_randomly(data_struct: ndarray) -> ndarray:
        """

        :param shape: a list representing shape to initialize randomly
        :return: a randomly initialized ndarray
        """
        if len(data_struct.shape) == 1:
            data_struct = [0.5 for i in data_struct]
        else:
            data_struct = [[np.random.randn() for i in data_struct[0] for j in range(len(data_struct))]]
        return data_struct


    @staticmethod
    def cost(predictions: ndarray, labels: np.array):
        """
        Returns the cost of a set or predictions

        :param predictions: a numpy array of per example predictions to compare with labels
        :param labels: A set of labels ordered in accordance to the training examples to compare against predictions
        """
        def populate_label_cost_matrix(labels: np.array) -> ndarray:
            """
            Returns a matrix of example labels

            :param labels: the true labels for the training examples
            :return label_cost_matrix: a matrix of examples labels
            """
            label_cost_matrix = np.zeros(shape=(len(labels), len(np.unique(labels))))
            for i, row in enumerate(label_cost_matrix):
                row[labels[i]] = 1
            return label_cost_matrix

        label_matrix = populate_label_cost_matrix(labels)
        error = predictions - label_matrix
        mean_squared_error = (np.sum(error ** 2) / len(error))
        derivative = (2 * error) / len(error)
        return mean_squared_error, derivative
