import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import os
from itertools import combinations

def jaccard_diversity(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    intersection = keys1 & keys2
    union = keys1 | keys2
    if not union:
        return -1  # Avoid division by zero; could also return 1 if both empty
    return len(intersection) / len(union)

def average_diversity(dict_list):
    n = len(dict_list)
    if n < 2:
        return -1
    diversities = [
        jaccard_diversity(d1, d2)
        for d1, d2 in combinations(dict_list, 2)
    ]
    return sum(diversities) / len(diversities)


def save_compiled(algo='ga'):
    random_files = sorted([x for x in os.listdir("./") if f"{algo}_result" in x and ".csv" in x],key=lambda x: int(x.replace(f"{algo}_result", "").replace(".csv","")))
    data = pd.DataFrame(np.zeros(shape=(10,7)),columns=["program",'Q#5','Q#10','Q#15','Q#20','Q#25','Q#29'])
    for file in random_files:
        Q = file.replace(f"{algo}_result", "").replace(".csv","")
        df = pd.read_csv(file)
        program_df = df.groupby(by=["Program","mutant"])
        for i,Gr in enumerate(program_df):
            P = Gr[0]
            group = Gr[1]
            name = P[0].split("_")[0]
            testcases = group["testcases"].values
            data.loc[i,"program"] = name+"_"+str(P[1])
            data.loc[i,f'Q#{Q}'] = testcases

    print(data)
    data.to_csv(f"{algo}_testcases.csv",index=False)

if __name__ == '__main__':
    # save_compiled(algo="1+1")
    program_qubit_diversity = pd.DataFrame(columns=["program", "qubit", "algorithm", "similarity"])
    for algo in ["rs", "hc","ga","1+1"]:
        df = pd.read_csv(f"{algo}_testcases.csv")
        for Q in ["5","10","15","20","25","29"]:
            for i in range(0,len(df)):
                testcases = eval(df.loc[i,f"Q#{Q}"])
                testcases = [x for x in testcases if x!={}]
                div = average_diversity(testcases)
                program_qubit_diversity.loc[len(program_qubit_diversity)] = [df.loc[i,"program"], int(Q), algo, div]

    df = program_qubit_diversity.groupby(by=["algorithm","qubit"]).mean(numeric_only=True).reset_index()
    plt.figure(figsize=(10, 6))
    pallet = sns.cubehelix_palette()
    plt.style.use('bmh')
    sns.pointplot(data=df, x='qubit', y='similarity', hue="algorithm",palette=pallet)  # or ci=95 for 95% CI

    plt.xlabel('Qubit', fontsize=18)
    plt.ylabel('Similarity', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(f"./graphs/similarity.png", bbox_inches='tight', dpi=300, transparent=True)