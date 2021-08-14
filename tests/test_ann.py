import pytest
from src.artificial_neural_network.ann import ANN
import pandas as pd


class TestAnn:
    """
    tester class for ann
    """
    @staticmethod
    def test_nn_xor_pattern():
        pass

    @staticmethod
    def test_nn_with_survey_data():
        pass

    @staticmethod
    def test_nn_synthetic_data():
        neural_net = ANN(3, 0.05, runtime_batch_size=10)
        neural_net.add_layer((3,5))
        neural_net.add_layer((5,6))
        neural_net.add_layer((6,3))
        pass

    @staticmethod
    def test_nn_with_incorrect_dimensionality():
        neural_net = ANN(5, 0.05, runtime_batch_size=10)
        df = pd.DataFrame({"f1": [1,2,3,4,5], "f2": [2,3,4,5,6], "f3": [3,4,5,6,7]})
        df2 = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [2, 3, 4, 5, 6], "f3": [3, 4, 5, 6, 7], "f4": [1, 2, 3, 4, 5],
                            "f5": [2, 3, 4, 5, 6]})

        def test_nn_with_incorrect_input_layer_dimensionality(df):
            try:
                neural_net.add_layer(weights=df.to_numpy())
                return False
            except AttributeError:
                return True

        def test_nn_with_incorrect_second_layer_dimensionality(df, df2):
            neural_net.add_layer(weights=df2.to_numpy())
            try:
                neural_net.add_layer(weights = df.to_numpy())
                return False
            except AttributeError:
                return True

        assert test_nn_with_incorrect_input_layer_dimensionality(df), "Input layer incorrect dimensionality issue not caught"
        assert test_nn_with_incorrect_second_layer_dimensionality(df, df2),  "Adding a second layer incorrect dimensionality issue not caught"

