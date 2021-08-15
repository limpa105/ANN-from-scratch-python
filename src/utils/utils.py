from numpy import ndarray
from numpy import random
from numpy import ndarray
import numpy as np


class Utils:
    @staticmethod
    def initialize_randomly(shape: list) -> ndarray:
        """

        :param shape: a list representing shape to initialize randomly
        :return: a randomly initialized ndarray
        """
        return random.rand(*shape)

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
            label_cost_matrix = np.zeros(len(labels),len(np.unique(labels)))
            for i, row in enumerate(label_cost_matrix):
                row[labels[i]] = 1
            return label_cost_matrix

        label_matrix = populate_label_cost_matrix(labels)
        error =  predictions - label_matrix
        squared_error = np.exp(np.log(error) + np.log(error))
        derivative = 2 * error
        return squared_error, derivative





class Matrix_Utils:
    pass
