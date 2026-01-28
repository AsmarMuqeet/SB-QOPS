import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import os

if __name__ == '__main__':
    random_files = sorted([x for x in os.listdir("./") if "ga_result" in x],key=lambda x: int(x.replace("ga_result", "").replace(".json","")))
    qubits = []
    name = []
    score = []
    for file in random_files:
        Q = file.replace("ga_result", "").replace(".json","")
        with open(file, 'r') as f:
            data = json.load(f)

        programs = data.keys()
        for program in programs:
            N = program.split("_")[0]
            mutation_score = []
            for run in data[program]:
                S = (len(run)/3)*100
                qubits.append(Q)
                name.append(N)
                score.append(S)

    df = pd.DataFrame({'qubits': qubits, 'name': name, 'score':score})
    fig, ax = plt.subplots(figsize=(15,7))
    sns.boxplot(data=df, x='qubits', y='score',hue='name',ax=ax,width=1)
    sns.move_legend(ax, "upper center",ncol=5,bbox_to_anchor=(0.5,1.14))
    plt.tight_layout()
    plt.show()