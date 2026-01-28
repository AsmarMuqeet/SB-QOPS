import json
import os
import pickle

import pandas as pd
import numpy as np

import QOPS as qops
from QOPS_test import load_program, get_mutants, get_compact_program_specification_Z
from tqdm.auto import tqdm
import sys


if __name__ == '__main__':

    QUBITS = sys.argv[1]
    Runs = 10
    algo = "1+1"

    ga_result = pd.DataFrame(columns=['Program',"mutant",'catch_avg','avg_fitness','testcases'])
    program_history = {}


    program_list = [x for x in os.listdir('benchmarkFilteration/benchmark2/') if x.split(".")[0].split("_")[-1] == QUBITS]
    for program_name in tqdm(program_list):
        circuit = load_program(program_name,"benchmarkFilteration/benchmark2")
        program_specification = get_compact_program_specification_Z(circuit, shots=20000)
        mutants = get_mutants(3,circuit,seed=1997)
        for mutant_index,mutant in enumerate(mutants):
            tester = qops.Circuit_Tester(CUT=mutant)
            tester.set_applicable_families_Z(program_specification)
            mutants_per_run = []
            fitness_per_run = []
            testcases_per_run = []
            history_per_run = []
            for runs in range(Runs):
                killed = 0
                pauli = {}
                fitness = 0
                for i in range(len(tester.applicable_families)):
                    best_function,testcase, history = tester.run_mealoneplusone(i, 80)
                    if best_function >0.1:
                        killed = 1
                        pauli = testcase
                        fitness = best_function
                        break
                mutants_per_run.append(killed)
                testcases_per_run.append(pauli)
                fitness_per_run.append(fitness)
                history_per_run.append(history)

            avg_score = np.mean(mutants_per_run)
            avg_fitness = np.mean(fitness_per_run)

            ga_result = pd.concat([pd.DataFrame([[program_name,mutant_index,avg_score,avg_fitness,json.dumps(testcases_per_run)]],columns=ga_result.columns),ga_result],ignore_index=True)
            program_history[program_name] = history_per_run
            ga_result.to_csv(f'{algo}_result{QUBITS}.csv',index=False)
            fp = open(f"{algo}_history{QUBITS}.pkl","wb")
            pickle.dump(program_history,fp)
            fp.close()