import unittest
from nn_lib import LinearLayer
from nn_lib import MultiLayerNetwork
from nn_lib import ReluLayer
from nn_lib import SigmoidLayer

class UnitTestCase(unittest.TestCase):
    def setUp(self):
        """Call before every test case"""
        self.network = MultiLayerNetwork(
            input_dim=4, neurons=[16, 2], activations=["relu", "sigmoid"]
        )
        
    def tearDown(self):
        """Call after every test case"""
        pass

    def testConstructAsExpectd(self):
        assert isinstance(self.network._layers[0], LinearLayer)
        assert isinstance(self.network._layers[1], ReluLayer)
        assert isinstance(self.network._layers[2], LinearLayer)
        assert isinstance(self.network._layers[3], SigmoidLayer)

if __name__ == "__main__":
    unittest.main() # run all tests