import os
import pickle as pkl
import gc
import time

import numpy as np
import random
import argparse

import theano
import theano.tensor as T
import train_dAs as trainer

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer

import classes.rfc as rfc
import classes.svm as svm
import classes.full_rf as fullrfc
import classes.nearest_neighbors as nearest_neighbors
import classes.da as dAutoencoder
import classes.tree as tree


def run(run_name='test', patient_counts=[100, 200, 500],
        da_patients=[1000], hidden_nodes=[1, 2, 4, 8],
        missing_data=[0]):
    # loop through patient files
    np.random.seed(seed=123)
    random.seed(123)

    overall_time = time.time()
    i = 0
    path = './data/' + run_name + '/'
    patients_path = path + 'patients'
    for file in os.listdir(patients_path):
        if file.endswith(".p"):
            for d in missing_data:
                run_start = time.time()
                i += 1
                scores = {}
                print(file, ' ', str(d))
                patients = pkl.load(open(patients_path + '/' + file, 'rb'))

                np.random.shuffle(patients)
                X = patients[:, :-1]
                y = patients[:, -1]

                if d > 0:
                    missing_vector = np.asarray(add_missing(patients, d))
                    X = np.array(X)
                    X[np.where(missing_vector == 0)] = 'NaN'
                    imp = Imputer(strategy='mean', axis=0)
                    X = imp.fit_transform(X)
                else:
                    missing_vector = None

                print(sum(y), len(y))

                dAs = {}
                for p in da_patients:
                    dAs[p] = {}
                    for n in hidden_nodes:
                        print(p, n)
                        dAs[p][n] = trainer.train_da(X[:p],
                                                     learning_rate=0.1,
                                                     coruption_rate=0.2,
                                                     batch_size=100,
                                                     training_epochs=1000,
                                                     n_hidden=n,
                                                     missing_data=missing_vector)

                for count in patient_counts:
                    scores[count] = classify(X[:count], y[:count], dAs)

                first_part = file.split('.p')[0]
                score_name = first_part + '_s.p'

                if len(missing_data) > 0:
                    pkl.dump(scores, open(path + 'scores/m' + str(d) + '_' +
                                          score_name, "wb"), protocol=2)
                else:
                    pkl.dump(scores, open(path + 'scores/' + score_name, "wb"),
                             protocol=2)
                print(scores)
                del patients
                run_end = time.time()
                print(i, ' run time:', str(run_end - run_start), ' total: ',
                      str(run_end - overall_time))
    print(i)


def perform_imputation(patients, missing_imputation):
    # mean imputation = 0
    if missing_imputation == 0:
        means = []
        for idx, col in enumerate(np.column_stack(patients)[:-1]):
            means.append(np.mean(col[np.nonzero(col)]))
        for idx, p in enumerate(patients):
            for jdx, col in enumerate(p[:-1]):
                if patients[idx][jdx] == 0:
                    patients[idx][jdx] = means[jdx]
        return patients

    # nearest neighbors imputation = 1
    elif missing_imputation == 1:
        print('nearest neighbors not yet implemented, performing mean')
        return perform_imputation(patients, 0)


def add_missing(patients, missing_data):
    zeros = (np.zeros(missing_data * 100))
    ones = (np.ones((1 - missing_data) * 100))
    one_zero = np.concatenate([zeros, ones])

    mv = []
    for p in patients:
        mv.append(np.random.choice(one_zero, len(p[:-1])))
    return mv


def classify(X, y, dAs):
    methods = {'rfc': rfc, 'fullrfc': fullrfc, 'svm': svm,
               'tree': tree, 'nearest_neighbors': nearest_neighbors}

    cv = StratifiedKFold(y, n_folds=10, random_state=123)
    prob_methods = {}
    scores = {}
    labels = []

    i_theano = T.dmatrix('i_theano')
    for i, (train, test) in enumerate(cv):
        for p_count in dAs.keys():
            for nodes in dAs[p_count].keys():
                get_hidden = dAs[p_count][nodes].get_hidden_values(i_theano)
                f = theano.function([i_theano], [get_hidden])
                train_set_x_hidden = f(X[train])[0]
                test_set_x_hidden = f(X[test])[0]

                for method in methods:
                    s_name = ('da_' + str(p_count) + '_' + str(nodes) + '_' +
                              str(method))
                    if method in prob_methods:
                        prob_methods[s_name].append(methods[method].classify(
                                                    train_set_x_hidden,
                                                    test_set_x_hidden,
                                                    y[train], y[test]))
                    else:
                        prob_methods[s_name] = []
                        prob_methods[s_name].append(methods[method].classify(
                                                    train_set_x_hidden,
                                                    test_set_x_hidden,
                                                    y[train], y[test]))

        for method in methods:
            if method in prob_methods:
                prob_methods[method].append(methods[method].classify(
                                            X[train], X[test],
                                            y[train], y[test]))
            else:
                prob_methods[method] = []
                prob_methods[method].append(methods[method].classify(
                                            X[train], X[test],
                                            y[train], y[test]))
        labels.append(y[test])

    for method in prob_methods:
        for x in list(range(len(prob_methods[method]))):
            if method in scores:
                scores[method].append(roc_auc_score(labels[x],
                                                    prob_methods[method][x],
                                                    average='macro',
                                                    sample_weight=None))
            else:
                scores[method] = [roc_auc_score(labels[x],
                                  prob_methods[method][x],
                                  average='macro', sample_weight=None)]
    gc.collect()
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", help="name the run")
    parser.add_argument("--patient_counts", nargs='*', default=400,
                        help="list of patient counts for classification")
    parser.add_argument("--da_patients", nargs='*', default=1000,
                        help="number of patients to train da")
    parser.add_argument("--hidden_nodes", nargs='*', default=2,
                        help="list of hidden nodes to train for")
    parser.add_argument("--missing_data", nargs='*', default=0,
                        help="list of percentage data missing")

    args = parser.parse_args()

    if args.patient_counts is None:
        args.patient_counts = [100, 200, 500]
    else:
        args.patient_counts = [int(x) for x in args.patient_counts]

    if args.da_patients is None:
        args.da_patients = [1000]
    else:
        args.da_patients = [int(x) for x in args.da_patients]

    if args.hidden_nodes is None:
        args.hidden_nodes = [1, 2, 4, 8]
    else:
        args.hidden_nodes = [int(x) for x in args.hidden_nodes]

    if args.missing_data is None:
        args.missing_data = [0]
    else:
        args.missing_data = [float(x) for x in args.missing_data]

    run(run_name=args.run_name,
        patient_counts=args.patient_counts,
        da_patients=args.da_patients,
        hidden_nodes=args.hidden_nodes,
        missing_data=args.missing_data)
