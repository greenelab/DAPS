import os
import sys
import shutil

import classes.sim as sim
import pickle as pkl
import argparse

DEFAULT_RUN_NAME = 'TEST'

def run(run_name=DEFAULT_RUN_NAME, trials=5, patient_count=2000,
        num_effects=[1, 2, 4, 8], observed_variables=[50], per_effect=[5],
        effect_mag=[1], sim_model=1, systematic_bias=0.1, input_variance=0.1,
        missing_data=0):
    i = 0
    patients_list = []

    for effects in num_effects:
        for ov in observed_variables:
            # for pe in per_effect:
            for mag in effect_mag:
                for trial in range(trials):
                    patients = sim.create_patients(patient_count=patient_count,
                                                   observed_variables=ov,
                                                   systematic_bias=systematic_bias,
                                                   input_variance=input_variance,
                                                   effects=effects,
                                                   per_effect=per_effect,
                                                   effect_mag=mag,
                                                   trial=trial,
                                                   sim_model=sim_model,
                                                   missing_data=missing_data)

                    file_name = ('./data/' + run_name + '/patients/' +
                                 str(sim_model) + '_' + str(effects) + '_' +
                                 str(per_effect) + '_' + str(mag) + '_' +
                                 str(ov) + '_' + str(trial) + '.p')

                    pkl.dump(patients, open(file_name, "wb"),
                             protocol=2)
                    del patients
                    print(i)
                    i += 1


def create_directories(run_name):
    data_path = 'data'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    run_path = data_path + '/' + run_name
    if os.path.exists(run_path):
        shutil.rmtree(run_path)

    os.makedirs(run_path)

    patients_path = run_path + '/patients'
    os.makedirs(patients_path)

    score_path = run_path + '/scores'
    os.makedirs(score_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # defaults are quick run
    parser.add_argument("--run_name", help="name the run")
    parser.add_argument("--trials", nargs='?', default=1,
                        help="number of independent replicates")
    parser.add_argument("--patient_count", nargs='?', default=100,
                        help="number of patient")
    parser.add_argument("--num_effects", nargs='*', help="""number of hidden
                        input effects (list of ints)""")
    parser.add_argument("--observed_variables", nargs='*', default=400,
                        help="observed variables per patient")
    parser.add_argument("--per_effect", nargs='?', default=5,
                        help="Number of clincial variables affected by hidden")
    parser.add_argument("--effect_mag", nargs='*',
                        help="Effect magnitude range (x input variance)")
    parser.add_argument("--sim_model", nargs='?', default=1,
                        help=""" - all or none hidden effects,
                        2 - continous (percent hidden = perecent case,
                        3 - rules based """)
    parser.add_argument("--systematic_bias", nargs='?', default=0.1,
                        help="""bias amount assessed randomly to half of the
                        patients (0 for none)""")
    parser.add_argument("--input_variance", nargs='?', default=0.1,
                        help="variance per variable")
    parser.add_argument("--missing_data", nargs='?', default=0,
                        help="Perecentage of data to be missing (randomly)")

    args = parser.parse_args()

    if args.observed_variables is None:
        args.observed_variables = [50, 100, 200]
    else:
        args.observed_variables = [int(x) for x in args.observed_variables]

    if args.num_effects is None:
        args.num_effects = [1, 2, 4, 8]
    else:
        args.num_effects = [int(x) for x in args.num_effects]

    if args.effect_mag is None:
        args.effect_mag = [1, 2, 4]
    else:
        args.effect_mag = [int(x) for x in args.effect_mag]

    if args.run_name is None:
        args.run_name = DEFAULT_RUN_NAME

    create_directories(args.run_name)
    target = open('./data/' + args.run_name + '/args.txt', 'w+')
    target.write(str(args))
    target.truncate()
    target.close()
    print(args)

    args.sim_model = int(args.sim_model)

    run(run_name=args.run_name,
        trials=int(args.trials),
        patient_count=int(args.patient_count),
        num_effects=args.num_effects,
        observed_variables=args.observed_variables,
        per_effect=int(args.per_effect),
        effect_mag=args.effect_mag,
        sim_model=int(args.sim_model),
        systematic_bias=float(args.systematic_bias),
        input_variance=float(args.input_variance),
        missing_data=float(args.missing_data))
