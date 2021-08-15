import pytest
from src.artificial_neural_network.ann import ANN
import pandas as pd
import numpy as np


class TestAnn:
    """
    tester class for ann
    """
    @staticmethod
    def test_nn_xor_pattern():
        XOR_RANGE = 2
        INPUT_SIZE = 95
        # TODO: VALIDATE SIZE PARAMETER IS_TUPLE, MAY NEED TO BE BROADCASTED ALONG 0TH DIMENSION
        data = np.random.randint(low = 0 , high = XOR_RANGE,size = (INPUT_SIZE))
        data2 = np.random.randint(low = 0 , high = XOR_RANGE,size = (INPUT_SIZE))
        y =  pd.Series([val1 and (not val2) or (val1 and (not val2)) for val1, val2 in zip(data,data2)])
        df = pd.DataFrame()
        X = pd.DataFrame.assign(df, col1 = data, col2 = data2)
        neural_net = ANN(2, 0.05)
        neural_net.add_layer((2,5))
        neural_net.add_layer((5,8))
        neural_net.add_layer((8,2))
        neural_net.add_layer((2,1))
        neural_net.train(epochs= 100, data = X, labels = y)
        predictions = neural_net.predict([[0,1], (1,0), [1,1], [0,0]])
        assert predictions == [1, 1, 0, 0] , "XOR function predicted incorrectly"


    @staticmethod
    def test_nn_synthetic_data(df):
        neural_net = ANN(3, 0.05, runtime_batch_size=10)
        neural_net.add_layer((3,5))
        neural_net.add_layer((5,6))
        neural_net.add_layer((6,3))
        data, labels = df.remove(columns = ["is_female"]), df["is_female"]
        neural_net.train(epochs=50, data=data, labels=labels)
        clear_male = [190, 70, 10]
        clear_female = [130, 55, 6.5]
        uncertain = [145, 65,  9]
        print("Predictions for clear male, clear female and uncertain are ", neural_net.predict([clear_male, clear_female, uncertain]))

    @staticmethod
    def test_nn_with_incorrect_dimensionality():
        neural_net = ANN(5, 0.05, runtime_batch_size=10)
        df = pd.DataFrame({"f1": [1,2,3,4,5], "f2": [2,3,4,5,6], "f3": [3,4,5,6,7]})
        df2 = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [2, 3, 4, 5, 6], "f3": [3, 4, 5, 6, 7], "f4": [1, 2, 3, 4, 5],
                            "f5": [2, 3, 4, 5, 6]})

        def test_nn_with_incorrect_input_layer_dimensionality(df):
            try:
                neural_net.add_layer(weights=df.to_numpy().T)
                return False
            except AttributeError:
                return True

        def test_nn_with_incorrect_second_layer_dimensionality(df, df2):
            neural_net.add_layer(weights=df2.to_numpy())
            try:
                neural_net.add_layer(weights = df.to_numpy().T)
                return False
            except AttributeError:
                return True

        assert test_nn_with_incorrect_input_layer_dimensionality(df), "Input layer incorrect dimensionality issue not caught"
        assert test_nn_with_incorrect_second_layer_dimensionality(df, df2),  "Adding a second layer incorrect dimensionality issue not caught"

if __name__ == "__main__":
    TestAnn.test_nn_with_incorrect_dimensionality()
    TestAnn.test_nn_synthetic_data()
    TestAnn.test_nn_xor_pattern()

