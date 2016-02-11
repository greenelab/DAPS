import numpy as np
import pickle as pkl
import random


def run(run_name='test1', patient_count=10, observed_variables=40,
        systematic_bias=0.1, input_variance=0.1, num_effects=[5],
        per_effect=5, effect_mag=2, trials=2, sim_model=1, semi_supervised=0,
        missing_data=0):
    """

    Create each set of the patients
    sim model 1 - simple (all or none)
    sim model 2 - all independent / single effect relevant
    sim model 3 - all independent / percentage based (percent of input effects)
    sim model 3 - all independent / complex rules based
    """

    # if semi supervised - create 10x as many patients
    if semi_supervised:
        sim_patients = patient_count * 10
    else:
        sim_patients = patient_count

    # generate mapping for each effect
    for effects in num_effects:
        for trial in range(trials):
            patients = create_patients(sim_patients, observed_variables,
                                       systematic_bias, input_variance,
                                       int(effects), per_effect, effect_mag,
                                       trial, sim_model, missing_data)
            save_patients(patients, run_name, effects, trial)


def create_patients(patient_count, observed_variables, systematic_bias,
                    input_variance, effects, per_effect, effect_mag,
                    trial, sim_model, missing_data):
    """

    Create and save a single run
    Saves Patients file using Effect Size and Trial as ids
    """
    random.seed(123)
    np.random.seed(seed=123)
    print("Model: ", sim_model)

    # @TODO check for balanced classes ***
    # map which clinical variables affected by hidden effect
    mappings = generate_mappings(observed_variables, effects, per_effect)

    # choose which patients have which effect + track (sim model matters here
    # to get adequate sample sizes)
    patient_status = generate_patient_status(patient_count, sim_model, effects)
    random.shuffle(patient_status)

    # generate data from patient status and mapping
    patients = generate_patients(mappings, patient_status, observed_variables,
                                 systematic_bias, input_variance, effects,
                                 effect_mag, sim_model)

    if systematic_bias:
        # @TODO this could be an argument / need to regression test this
        bias_cohort = 0.5
        patients = add_bias(np.array(patients), bias_cohort, systematic_bias)

    if missing_data > 0:
        patients = add_missing(patients, missing_data)

    return scale_patients(np.array(patients))


def add_missing(patients, missing_data):
    zeros = (np.zeros(missing_data * 100))
    ones = (np.ones((1 - missing_data) * 100))
    one_zero = np.concatenate([zeros, ones])

    for i, p in enumerate(patients):
        choose_missing = np.random.choice(one_zero, len(p[:-1]))
        patients[i, :-1] = patients[i, :-1] * choose_missing

    return patients


def add_bias(patients, bias_cohort, bias_percent):
    for i, p in enumerate(patients):
        if i < (len(patients) * bias_cohort):
            patients[i, :-1] = patients[i, :-1] + bias_percent
    return patients


def generate_patients(mappings, patient_status, observed_variables,
                      systematic_bias, input_variance, effects, effect_mag,
                      sim_model):

    # get mean shift per hidden
    # effect_mag = ((np.random.rand(effects * 2) - 1) * input_variance)
    effect_mag = ((np.random.rand(effects) * 2 - 1) *
                  input_variance * effect_mag)
    patients = []

    # for each patient simulate data, shifting data
    for p in patient_status:
        patient = []
        means = [1] * observed_variables

        for idx, mapping in enumerate(mappings):
            # only shift mean if patient has this effect
            if (p[idx]):
                for e in mapping:
                    means[e] = means[e] + effect_mag[idx]

        for obs in range(observed_variables):
            patient.append(np.random.normal(loc=means[obs],
                                            scale=(input_variance * 3)))

        patient.append(p[-1])
        patients.append(patient)
    return patients


def generate_patient_status(patient_count, sim_model, effects):
    """

    Create cohorts of patients
    Sim model 1 - turn effects on for 50 percent of patients and off
    for the other 50 percent
    """
    case_control_split = int(0.5 * patient_count)

    p_status_list = []
    if sim_model == 1:
        for p in range(patient_count):
            patient = []
            if p < case_control_split:
                for e in range(effects):
                    patient.append(1)
                patient.append(1)
            else:
                for e in range(effects):
                    patient.append(0)
                patient.append(0)
            p_status_list.append(patient)

    elif sim_model == 2:
        case_count = 0
        control_count = 0
        cases = []
        controls = []

        while (case_count <= case_control_split or
               control_count <= case_control_split):

            patient = np.random.randint(2, size=effects).tolist()
            patient_bin = int(''.join(str(e) for e in patient), 2)

            # if effect_map[patient_bin]:
            if patient_bin % 2:
                patient.append(1)
                case_count += 1
                cases.append(patient)
            else:
                patient.append(0)
                control_count += 1
                controls.append(patient)

        p_status_list = (cases[:case_control_split] +
                         controls[:case_control_split])

    elif sim_model == 3:
        case_count = 0
        control_count = 0
        cases = []
        controls = []

        while (case_count <= case_control_split or
               control_count <= case_control_split):
            patient = np.random.randint(2, size=effects).tolist()
            if np.random.rand() < np.mean(patient):
                patient.append(1)
                case_count += 1
                cases.append(patient)
            else:
                control_count += 1
                patient.append(0)
                controls.append(patient)

        p_status_list = (cases[:case_control_split] +
                         controls[:case_control_split])

    elif sim_model == 4:
        case_count = 0
        control_count = 0
        cases = []
        controls = []

        while (case_count <= case_control_split or
               control_count <= case_control_split):

            patient = np.random.randint(2, size=effects).tolist()
            patient_bin = np.sum(patient)

            # if effect_map[patient_bin]:
            if patient_bin % 2:
                patient.append(1)
                case_count += 1
                cases.append(patient)
            else:
                patient.append(0)
                control_count += 1
                controls.append(patient)

        p_status_list = (cases[:case_control_split] +
                         controls[:case_control_split])

    elif sim_model == 5:
        case_count = 0
        control_count = 0
        cases = []
        case_2 = []
        controls = []

        while (case_count <= case_control_split or
               control_count <= case_control_split):

            patient = np.random.randint(2, size=effects).tolist()
            patient_bin = int(''.join(str(e) for e in patient), 2)

            # if effect_map[patient_bin]:
            if patient_bin % 2:
                patient.append(1)
                case_count += 1
                cases.append(patient)
            else:
                patient.append(0)
                control_count += 1
                controls.append(patient)

        for i in range(case_control_split):
            patient = []
            for e in range(effects):
                patient.append(1)
            patient.append(1)
            case_2.append(patient)

        p_status_list = (cases[:case_control_split] +
                         controls[:case_control_split] +
                         case_2)

    print(len(p_status_list))
    return p_status_list


def generate_mappings(observed_variables, run_effects, per_effect):
    run_effect_mapping = []
    for x in range(run_effects):
        run_effect_mapping.append(np.random.randint(
                                  0, observed_variables - 1,
                                  size=per_effect).tolist())
    return run_effect_mapping


def scale_patients(patients):
    for idx, c in enumerate(patients.T[:-1, :]):
        c = (c - min(c)) / (max(c) - min(c))
        patients.T[idx] = c
    return patients


def save_patients(p, run_name, effects, idx):
    pkl.dump(p, open("./data/" + run_name + "/patients/" +
             str(effects) + "_" + str(idx) + ".p", "wb"), protocol=2)

# run only this module
if __name__ == "__main__":
    run(run_name='test1')
