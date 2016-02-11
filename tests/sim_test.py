import unittest
import sim
import numpy as np


class SimTestMethods(unittest.TestCase):

    def test_createPatientsShape(self):
        patients = sim.create_patients(patient_count=500, observations=50)
        self.assertEqual(patients.shape[0], 500)
        # 51 because of class
        self.assertEqual(patients.shape[1], 51)

    def test_create_patients_ratio(self):
        patients = sim.create_patients(patient_count=500,
                                       case_control_ratio=0.5)
        self.assertEqual(sum(patients[:, -1]), 250)

    def test_scale_patients(self):
        patients = sim.create_patients()
        self.assertTrue(np.max(patients) <= 1)

    # @TODO - not sure how to test bias

if __name__ == '__main__':
    unittest.main()
