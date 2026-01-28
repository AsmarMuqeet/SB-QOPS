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
    Runs = 1
    algo = "IBM1+1"

    ga_result = pd.DataFrame(columns=['Program',"mutant",'catch_avg','avg_fitness','testcases'])
    program_history = {}


    program_list = sorted([x for x in os.listdir('benchmarkFilteration/benchmark2/') if x.split(".")[0].split("_")[-1] == QUBITS])
    for program_name in tqdm(program_list[0:1]):
        print(program_name)
        circuit = load_program(program_name,"benchmarkFilteration/benchmark2")
        program_specification = get_compact_program_specification_Z(circuit, shots=10000)
        mutants = [circuit]
        mtname = {0:"eq",1:"f1",2:"f2",3:"f3"}
        mutants.extend(get_mutants(3,circuit,seed=1997))
        for mutant_index,mutant in enumerate(mutants):
            tester = qops.Circuit_Tester_IBM_ZNE(CUT=mutant,simulator_type="Noise",device_name="ibm_brisbane", shots=10000)
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
                    best_function,testcase, history = tester.run_customoneplusone(i, 20)
                    fitness = best_function
                    #if best_function >0.1:
                    #killed = 1
                    pauli = testcase
                    #break
                mutants_per_run.append(killed)
                testcases_per_run.append(pauli)
                fitness_per_run.append(fitness)
                history_per_run.append(history)

            avg_score = json.dumps(mutants_per_run)
            avg_fitness = json.dumps(fitness_per_run)

            ga_result = pd.concat([pd.DataFrame([[program_name,mtname[mutant_index],avg_score,avg_fitness,json.dumps(testcases_per_run)]],columns=ga_result.columns),ga_result],ignore_index=True)
            program_history[program_name] = history_per_run
            ga_result.to_csv(f'{algo}_result{QUBITS}.csv',index=False)
            fp = open(f"{algo}_history{QUBITS}.pkl","wb")
            pickle.dump(program_history,fp)
            fp.close()