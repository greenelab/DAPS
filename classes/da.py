import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class dA(object):
    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784,
                 n_hidden=500, W=None, bhid=None, bvis=None,
                 missing_data=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing
        if input is None:
            # use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.md = missing_data
        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_weights(self):
        return self.W.get_value()

    def get_cost(self):
        return self.cost

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        # L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        if self.md:
            # can use cross entropy or mse
            L = - T.sum(self.x * T.log(z) * self.md + (1 - self.x) * T.log(1 - z) * self.md, axis=1)
            #L_mat = (self.x - z) * self.md
            #L = T.sum(pow(abs(L_mat), 2), axis=1)
        else:
            # can use cross entropy or mse
            #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
            L = T.sum(pow(abs(self.x - z), 2), axis=1)

        cost = T.mean(L)

        # compute the gradients of the cost of the `dA`
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        self.cost = cost
        return (cost, updates)
