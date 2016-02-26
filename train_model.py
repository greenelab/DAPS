import os
import pickle

import numpy as np
import argparse

import train_dAs as trainer


def run(file, epochs, hidden_nodes):
    print(file, ' ', epochs, ' ', hidden_nodes)
    with open(file, 'rb') as f:
        patients = pickle.load(f)
    print(patients.shape)

    X = patients[:, :-1]
    y = patients[:, -1].astype(int)

    dA = {}

    for hn in hidden_nodes:
        for e in epochs:
            name = str(hn) + '_' + str(e)
            dA[name] = trainer.train_da(X, learning_rate=0.1,
                                        coruption_rate=0.2,
                                        batch_size=10,
                                        training_epochs=e,
                                        n_hidden=hn)
    save_run(file, dA)


def save_run(file, dA):
    for key, value in dA.items():
        hn, epoch = key.split('_')
        model = file.split('/')[2]

        file_name = './data/' + model + '/trained/' + hn + '_' + epoch + '.p'
        f = open(file_name, 'wb')
        pickle.dump(value, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="model number (folder name)")
    parser.add_argument("--run_name", help="id for run (file name)")
    parser.add_argument("--training_epochs", nargs='*', default=[10],
                        help="list of training epochs to save")
    parser.add_argument("--hidden_nodes", nargs='*', default=[2],
                        help="list of hidden nodes to save")

    args = parser.parse_args()

    if args.model_name is None:
        raise Exception('Model Name must be set')
    if args.run_name is None:
        raise Exception('Run name must be set')
    if args.training_epochs is None:
        args.training_epochs = [10]
    else:
        args.training_epochs = [int(x) for x in args.training_epochs]
    if args.hidden_nodes is None:
        args.hidden_nodes = [2]
    else:
        args.hidden_nodes = [int(x) for x in args.hidden_nodes]

    file_name = ('./data/' + args.model_name + '/patients/' +
                 args.run_name + '.p')
    print(file_name)
    run(file=file_name,
        epochs=args.training_epochs,
        hidden_nodes=args.hidden_nodes)
