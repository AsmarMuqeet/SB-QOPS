import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import os

if __name__ == '__main__':
    rs_history = sorted([x for x in os.listdir("./") if "rs_history" in x and ".pkl" in x],key=lambda x: int(x.replace("rs_history", "").replace(".pkl","")))
    with open(rs_history[0],'rb') as file:
        history = pickle.load(file)

    print(len(history['ghz_indep_qiskit_5.qasm']))