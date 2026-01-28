import json
import os

import numpy as np
import pandas as pd
import sys
sys.path.append("..")

import QOPS as qops
from QOPS_test import load_program, get_compact_program_specification_Z, get_mutants
from tqdm.auto import tqdm


if __name__ == '__main__':

    QUBITS = "5"
    SHOTS = 4000
    AL = "HC"

    Epochs = [10,20,25,40,50,60,70,80,90,100]
    Pop_size = [10,20,30,40,50]
    total_runs = 3

    result_epoch = []
    result_pop = []
    result_score = []
    for epoch in tqdm(Epochs):
        for pop in Pop_size:
            program_list = [x for x in os.listdir('../benchmarkFilteration/benchmark2/') if x.split(".")[0].split("_")[-1] == QUBITS]
            mutants_per_run = []
            for runs in range(total_runs):
                mutant_number = 0
                for program_name in program_list:
                    circuit = load_program(program_name,"../benchmarkFilteration/benchmark2")
                    program_specification = get_compact_program_specification_Z(circuit, shots=SHOTS)
                    mutants = get_mutants(3,circuit,seed=1997)
                    for mutant in mutants:
                        tester = qops.Circuit_Tester(CUT=mutant,shots=SHOTS)
                        tester.set_applicable_families_Z(program_specification)
                        for i in range(len(tester.applicable_families)):
                            best_function, testcase = tester.run_mealhillclimbing(i,epoch,pop)
                            if best_function> 0.1:
                                mutant_number+=1
                                break


                mutants_per_run.append(mutant_number/30)

            score = np.mean(mutants_per_run)

            result_epoch.append(epoch)
            result_pop.append(pop)
            result_score.append(score)

    data = pd.DataFrame()
    data['epoch'] = result_epoch
    data['pop'] = result_pop
    data['score'] = result_score
    data.to_csv(f'{AL}_tunning.csv',index=False)