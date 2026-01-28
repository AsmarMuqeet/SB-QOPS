import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import os

if __name__ == '__main__':
    algo = 'ga'
    random_files = sorted([x for x in os.listdir("./") if f"{algo}_result" in x and ".csv" in x],key=lambda x: int(x.replace(f"{algo}_result", "").replace(".csv","")))
    data = pd.DataFrame(np.zeros(shape=(10,7)),columns=["program",'Q#5','Q#10','Q#15','Q#20','Q#25','Q#29'])
    for file in random_files:
        Q = file.replace(f"{algo}_result", "").replace(".csv","")
        df = pd.read_csv(file)
        program_df = df.groupby(by=["Program"])
        data['program'] = data['program'].astype(str)
        for i,Gr in enumerate(program_df):
            P = Gr[0]
            group = Gr[1]
            name = P[0].split("_")[0]
            percent = group["catch_avg"].mean()
            data.loc[i,"program"] = name
            data.loc[i,f'Q#{Q}'] = percent

    print(data)
    data.to_csv(f"{algo}_compiled.csv",index=False)