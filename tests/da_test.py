import unittest
import da as dAutoencoder
import numpy as np

import theano
import theano.tensor as T


class DaTestMethods(unittest.TestCase):

    # @TODO not sure how to test this further
    def test_daCost(self):
        X = [[0, 1, 0, 1, 1, 1]] * 100
        da = dAutoencoder.train_da(X)
        self.assertTrue(da.trained_cost < 0.05)

if __name__ == '__main__':
    unittest.main()
