import unittest
import numpy as np

import classes.sim as sim
import create_patients as creator
import classify_patients as classifier
import train_dAs as trainer


class SimTestMethods(unittest.TestCase):

    def testDaRun(self):
        p = 100
        hn = 2

        patients = sim.create_patients(patient_count=p,
                                       observed_variables=100,
                                       systematic_bias=0.1,
                                       input_variance=0.1,
                                       effects=4,
                                       per_effect=5,
                                       effect_mag=5,
                                       trial=1,
                                       sim_model=1,
                                       missing_data=0)
        X = patients[:, :-1]
        y = patients[:, -1]

        dAs = {}
        dAs[p] = {}
        dAs[p][hn] = trainer.train_da(X,
                                      learning_rate=0.1,
                                      coruption_rate=0.2,
                                      batch_size=10,
                                      training_epochs=1000,
                                      n_hidden=hn,
                                      missing_data=None)

        self.assertTrue(str(dAs[p][hn].trained_cost)[:7] == str(3.78843))

        scores = classifier.classify(X, y, dAs)
        print(scores)

        da_rfc_scores = [1.0, 1.0, 1.0, 0.90000000000000002, 1.0, 1.0,
                         0.96000000000000008, 1.0, 0.78000000000000003, 1.0]
        rfc_score_name = 'da_' + str(p) + '_' + str(hn) + '_rfc'
        self.assertTrue(scores[rfc_score_name] == da_rfc_scores)


if __name__ == '__main__':
    unittest.main()
