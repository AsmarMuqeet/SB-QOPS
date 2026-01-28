# SB-QOPS: Search-Based Quantum Program Testing via Commuting Pauli String
This repository contains the code necessary to reproduce the results of SB-QOPS, excluding execution scripts on real computers of IBM, IQM, and Quantinuum.
The processed results and raw results for all executions (Simulations and real computer executions are provided in the repository)
# Installation
The repository is only tested in a Linux environment since Qiskit AER GPU is only supported in linux

### Dependencies

Anaconda Python distribution is required [here](https://www.anaconda.com/products/distribution):

Steps:

    1. Clone the repository
    2. cd SB-QOPS
    3  conda env create -f environment.yml
    4. conda activate sbqops 
	

# Results:
The results folder contains the scripts and necessary files for producing the results of RQ1 and RQ3.
The noise_result folder contains the scripts and files needed to produce the results for RQ2.
Proprietary executions such as qnexus results and produced files are in the producedfile folder.
# Evaluate new Circuit:
### To test new circuits

``` python

import QOPS as qops

if __name__ == '__main__':

    QUBITS = 29 # number of qubits

    ga_result = pd.DataFrame(columns=['Program',"mutant",'catch_avg','avg_fitness','testcases'])
    circuit = # qiskit circuit without measurements
    program_specification = #compact program specification in the form {"paulistring eg zzzz": {"bitstrings":count}}
    tester = qops.Circuit_Tester(CUT=circuit)
    tester.set_applicable_families_Z(program_specification)
    for i in range(len(tester.applicable_families)):
        best_function,testcase, history = tester.run_mealoneplusone(i, 80) # change algorithm here
            if best_function > 0.1: # tolerance threshold
                killed = 1
                pauli = testcase
                fitness = best_function
                break
```
