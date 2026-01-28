from itertools import combinations
import numpy as np
import pandas as pd
import os
from cliffs_delta import cliffs_delta

from scipy.stats import wilcoxon

if __name__ == '__main__':
    algos = ['rs','ga','hc','1+1']
    data = {}
    for algo in algos:
        data[algo] = []

    for algo in algos:
        random_files = sorted([x for x in os.listdir("./") if f"{algo}_result" in x and ".csv" in x],key=lambda x: int(x.replace(f"{algo}_result", "").replace(".csv","")))
        for file in random_files:
            Q = file.replace(f"{algo}_result", "").replace(".csv", "")
            if Q!="39":
                df = pd.read_csv(file)
                df.sort_values(by=["Program","mutant"],inplace=True)
                for i in range(df.shape[0]):
                    catch = df.loc[i,"catch_avg"]
                    data[algo].append(catch)

    # for k in data.keys():
    #     print(k, len(data[k]))
    df = pd.DataFrame(data)

    result_df = pd.DataFrame(columns=["Algorithm A","Algorithm B","p-value","effect_size","interpretation"])

    # Loop through all pairs of columns
    for col1, col2 in combinations(df.columns, 2):
        stat, p = wilcoxon(df[col1], df[col2])
        d, res = cliffs_delta(df[col1], df[col2])

        result_df.loc[len(result_df)] = [col1, col2, p, d, res]

    result_df.to_csv("statistical_test.csv",index=False)
    print(result_df)