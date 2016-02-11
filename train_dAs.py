import classes.da

import os
import sys
import timeit

import numpy as np
import pickle as pkl

import classes.da as dAutoencoder

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


def run(run_name='Test1', hidden_nodes=[2], training_epochs=100,
        semi_supervised=0, indices=None):
    path = 'data/' + run_name + '/patients'
    cost_dict = {}

    for file in os.listdir(path):
        if file.endswith(".p"):
            patients = pkl.load(open(path + '/' + file, 'rb'))

            if indices:
                X = patients[indices, :-1]
            else:
                X = patients[:, :-1]

            effects, trial = file.split('.')[0].split('_')

            for hn in hidden_nodes:
                da = train_da(X, training_epochs=training_epochs,
                              n_hidden=hn)
                save_da(da, run_name, effects, trial, hn, indices)

                if effects in cost_dict:
                    if hn in cost_dict[effects]:
                        cost_dict[effects][hn].append(da.cost_list)
                    else:
                        cost_dict[effects][hn] = [da.cost_list]
                else:
                    cost_dict[effects] = {}
                    cost_dict[effects][hn] = [da.cost_list]

    target = open('./data/' + run_name + '/costs.txt', 'w+')
    target.write(str(cost_dict))
    target.truncate()
    target.close()


def save_da(da, run_name, effects, trial, hn, indices):
    if indices:
        print('save run:', run_name, ' effects: ', effects, ' trial:', trial,
              ' hidden nodes: ', hn, 'indices len: ', len(indices))
    else:
        print('save run:', run_name, ' effects: ', effects, ' trial:', trial,
              ' hidden nodes: ', hn)

    if indices:
        pkl.dump(da, open('data/' + run_name + "/trained_da/" + str(effects) +
                 "_" + str(trial) + "_" + str(hn) + "_i.p", "wb"), protocol=2)
    else:
        pkl.dump(da, open('data/' + run_name + "/trained_da/" + str(effects) +
                 "_" + str(trial) + "_" + str(hn) + ".p", "wb"), protocol=2)


def train_da(X, missing_data=None, learning_rate=0.1, coruption_rate=0.2,
             batch_size=10, training_epochs=100, n_visible=1000, n_hidden=2):
    train_set_x = theano.shared(np.asarray(X, dtype=theano.config.floatX),
                                borrow=True)

    if missing_data is not None:
        md = theano.shared(np.asarray(missing_data,
                           dtype=theano.config.floatX), borrow=True)
    else:
        md = None

    features = train_set_x.get_value(borrow=True).shape[1]
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] /
                          batch_size)

    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dAutoencoder.dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        missing_data=md,
        n_visible=features,
        n_hidden=n_hidden
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.2,
        learning_rate=learning_rate
    )

    if missing_data is not None:
        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                md: md[index * batch_size: (index + 1) * batch_size]

            }
        )
    else:
        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

    start_time = timeit.default_timer()

    cost_list = []
    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        cost = np.mean(c)
        cost_list.append(cost)
        if (epoch + 1) % 100 == 0:
            print('Training epoch ' + str(epoch + 1) + ': ' + str(np.mean(c)))

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    da.trained_cost = cost
    da.cost_list = cost_list
    return da
