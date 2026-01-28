import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import os
from mealpy.utils.history import History
from mealpy.utils.agent import Agent
from PIL import Image, ImageDraw, ImageFont

def get_rs_data():
    rs_history = sorted([x for x in os.listdir("./") if "rs_history" in x and ".pkl" in x],key=lambda x: int(x.replace("rs_history", "").replace(".pkl","")))
    Program_Qubit = []
    for files in rs_history:
        with open(files,'rb') as file:
            history = pickle.load(file)

        Program_Qubit.extend([(x.split("_")[0],x.split("_")[-1].replace(".qasm","")) for x in sorted(history.keys(),
                                                                                                     key=lambda x: x.split("_")[0])])

    rs_data = {}
    for files in rs_history:
        with open(files,'rb') as file:
            history = pickle.load(file)
            selected_q = list(history.keys())[0].split("_")[-1].replace(".qasm","")
        for program_name,qubit in Program_Qubit:
            if qubit==selected_q:
                runs = history[f"{program_name}_indep_qiskit_{qubit}.qasm"]
                data = []
                for i in range(len(runs)):
                    fit = runs[i]
                    best_individual = []
                    for j in range(0,len(fit),40):
                        temp = fit[j:j+40]
                        best = max(temp)
                        best_individual.append(best)
                    data.append(best_individual)
                rs_data[(program_name,qubit)] = np.array(data)

    return rs_data


def get_search_data(algo):
    random.seed(42)
    rs_history = sorted([x for x in os.listdir("./") if f"{algo}_history" in x and ".pkl" in x],key=lambda x: int(x.replace(f"{algo}_history", "").replace(".pkl","")))
    Program_Qubit = []
    for files in rs_history:
        with open(files,'rb') as file:
            history = pickle.load(file)

        Program_Qubit.extend([(x.split("_")[0],x.split("_")[-1].replace(".qasm","")) for x in sorted(history.keys(),
                                                                                                     key=lambda x: x.split("_")[0])])

    rs_data = {}
    for files in rs_history:
        with open(files,'rb') as file:
            history = pickle.load(file)
            selected_q = list(history.keys())[0].split("_")[-1].replace(".qasm","")
        for program_name,qubit in Program_Qubit:
            if qubit==selected_q:
                runs = history[f"{program_name}_indep_qiskit_{qubit}.qasm"]

                data = []
                for i in range(len(runs)):
                    fit = runs[i]
                    best_individual = [x.target.fitness for x in fit.list_current_best]

                    data.append(best_individual)
                rs_data[(program_name,qubit)] = np.array(data)

    return rs_data

def get_one_data():
    random.seed(42)
    algo="1+1"
    rs_history = sorted([x for x in os.listdir("./") if f"{algo}_history" in x and ".pkl" in x],key=lambda x: int(x.replace(f"{algo}_history", "").replace(".pkl","")))
    Program_Qubit = []
    for files in rs_history:
        with open(files,'rb') as file:
            history = pickle.load(file)

        Program_Qubit.extend([(x.split("_")[0],x.split("_")[-1].replace(".qasm","")) for x in sorted(history.keys(),
                                                                                                     key=lambda x: x.split("_")[0])])

    rs_data = {}
    for files in rs_history:
        with open(files,'rb') as file:
            history = pickle.load(file)
            selected_q = list(history.keys())[0].split("_")[-1].replace(".qasm","")
        for program_name,qubit in Program_Qubit:
            if qubit==selected_q:
                runs = history[f"{program_name}_indep_qiskit_{qubit}.qasm"]
                data = []
                for i in range(len(runs)):
                    fit = runs[i]
                    best_individual = []
                    for j in range(0,len(fit.list_current_best_fit),8):
                        temp = fit.list_current_best_fit[j:j+8]
                        best = max(temp)
                        best_individual.append(best)

                    data.append(best_individual)
                rs_data[(program_name,qubit)] = np.array(data)
    return rs_data


def create_individual_graphs():
    rs_data = get_rs_data()
    ga_data = get_search_data("ga")
    hc_data = get_search_data("hc")
    one_data = get_one_data()
    for program_name,qubit in rs_data.keys():
        if qubit!='29':
            df = pd.DataFrame(rs_data[(program_name,qubit)])
            df['Run'] = df.index
            rs_df_long = df.melt(id_vars='Run', var_name='Generation', value_name='Fitness')
            rs_df_long['Generation'] = rs_df_long['Generation'].astype(int)+1
            rs_df_long['Algorithm'] = "RS"

            df = pd.DataFrame(ga_data[(program_name, qubit)])
            df['Run'] = df.index
            ga_df_long = df.melt(id_vars='Run', var_name='Generation', value_name='Fitness')
            ga_df_long['Generation'] = ga_df_long['Generation'].astype(int) + 1
            ga_df_long['Algorithm'] = "GA"
            ga_df_long.loc[ga_df_long['Generation']==1,"Fitness"] = rs_df_long.loc[rs_df_long['Generation']==1,"Fitness"]

            df = pd.DataFrame(hc_data[(program_name, qubit)])
            df['Run'] = df.index
            hc_df_long = df.melt(id_vars='Run', var_name='Generation', value_name='Fitness')
            hc_df_long['Generation'] = hc_df_long['Generation'].astype(int) + 1
            hc_df_long['Algorithm'] = "HC"
            hc_df_long.loc[hc_df_long['Generation'] == 1,"Fitness"] = rs_df_long.loc[rs_df_long['Generation'] == 1,"Fitness"]

            df = pd.DataFrame(one_data[(program_name, qubit)])
            df['Run'] = df.index
            one_df_long = df.melt(id_vars='Run', var_name='Generation', value_name='Fitness')
            one_df_long['Generation'] = one_df_long['Generation'].astype(int) + 1
            one_df_long['Algorithm'] = "1+1"
            one_df_long.loc[one_df_long['Generation'] == 1, "Fitness"] = rs_df_long.loc[rs_df_long['Generation'] == 1, "Fitness"]

            DATA = pd.concat([rs_df_long, ga_df_long, hc_df_long, one_df_long],ignore_index=True)
            plt.figure(figsize=(10, 6))
            pallet = sns.cubehelix_palette()
            plt.style.use('bmh')
            sns.pointplot(data=DATA, x='Generation', y='Fitness', errorbar='ci',hue="Algorithm",estimator='median',palette=pallet)  # or ci=95 for 95% CI

            #plt.title('Best Fitness per Generation (10 Runs)')
            plt.xlabel('Generation')
            plt.ylabel('Best Fitness')
            #plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"./graphs/{program_name}_{qubit}.png", bbox_inches='tight',dpi=300,transparent=True)

def create_fitness_generation_graphs():
    rs_data = get_rs_data()
    ga_data = get_search_data("ga")
    hc_data = get_search_data("hc")
    one_data = get_one_data()
    merged = []
    for program_name,qubit in rs_data.keys():
        if qubit!='29':
            df = pd.DataFrame(rs_data[(program_name,qubit)])
            df['Run'] = df.index
            rs_df_long = df.melt(id_vars='Run', var_name='Generation', value_name='Fitness')
            rs_df_long['Generation'] = rs_df_long['Generation'].astype(int)+1
            rs_df_long['Algorithm'] = "RS"

            df = pd.DataFrame(ga_data[(program_name, qubit)])
            df['Run'] = df.index
            ga_df_long = df.melt(id_vars='Run', var_name='Generation', value_name='Fitness')
            ga_df_long['Generation'] = ga_df_long['Generation'].astype(int) + 1
            ga_df_long['Algorithm'] = "GA"
            ga_df_long.loc[ga_df_long['Generation'] == 1, "Fitness"] = rs_df_long.loc[
                rs_df_long['Generation'] == 1, "Fitness"]

            df = pd.DataFrame(hc_data[(program_name, qubit)])
            df['Run'] = df.index
            hc_df_long = df.melt(id_vars='Run', var_name='Generation', value_name='Fitness')
            hc_df_long['Generation'] = hc_df_long['Generation'].astype(int) + 1
            hc_df_long['Algorithm'] = "HC"
            hc_df_long.loc[hc_df_long['Generation'] == 1, "Fitness"] = rs_df_long.loc[
                rs_df_long['Generation'] == 1, "Fitness"]

            df = pd.DataFrame(one_data[(program_name, qubit)])
            df['Run'] = df.index
            one_df_long = df.melt(id_vars='Run', var_name='Generation', value_name='Fitness')
            one_df_long['Generation'] = one_df_long['Generation'].astype(int) + 1
            one_df_long['Algorithm'] = "1+1"
            one_df_long.loc[one_df_long['Generation'] == 1, "Fitness"] = rs_df_long.loc[
                rs_df_long['Generation'] == 1, "Fitness"]

            DATA = pd.concat([rs_df_long, ga_df_long, hc_df_long, one_df_long],ignore_index=True)
            DATA["Program"] = program_name
            DATA["Qubit"] = qubit
            merged.append(DATA.copy())
    merged = pd.concat(merged, ignore_index=True)
    df_group = merged.groupby(["Run","Generation","Algorithm"])
    df_fit = df_group.mean(numeric_only=True)
    df_fit.reset_index(inplace=True)

    plt.figure(figsize=(10, 6))
    pallet = sns.cubehelix_palette()
    plt.style.use('bmh')
    sns.pointplot(data=df_fit, x='Generation', y='Fitness', errorbar='sd', hue="Algorithm", estimator='mean',
                  palette=pallet)  # or ci=95 for 95% CI

    plt.title('Best Fitness per Generation (10 Runs)')
    plt.xlabel('Generation')
    plt.ylabel('Avg Best Fitness')
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./graphs/generations.png", bbox_inches='tight', dpi=300, transparent=True)
    plt.show()

def create_fitness_generation_qubit_graphs():
    rs_data = get_rs_data()
    ga_data = get_search_data("ga")
    hc_data = get_search_data("hc")
    one_data = get_one_data()
    merged = []
    for program_name,qubit in rs_data.keys():
        # if qubit!='29':
        df = pd.DataFrame(rs_data[(program_name,qubit)])
        df['Run'] = df.index
        rs_df_long = df.melt(id_vars='Run', var_name='Generation', value_name='Fitness')
        rs_df_long['Generation'] = rs_df_long['Generation'].astype(int)+1
        rs_df_long['Algorithm'] = "RS"

        df = pd.DataFrame(ga_data[(program_name, qubit)])
        df['Run'] = df.index
        ga_df_long = df.melt(id_vars='Run', var_name='Generation', value_name='Fitness')
        ga_df_long['Generation'] = ga_df_long['Generation'].astype(int) + 1
        ga_df_long['Algorithm'] = "GA"
        ga_df_long.loc[ga_df_long['Generation'] == 1, "Fitness"] = rs_df_long.loc[
            rs_df_long['Generation'] == 1, "Fitness"]

        df = pd.DataFrame(hc_data[(program_name, qubit)])
        df['Run'] = df.index
        hc_df_long = df.melt(id_vars='Run', var_name='Generation', value_name='Fitness')
        hc_df_long['Generation'] = hc_df_long['Generation'].astype(int) + 1
        hc_df_long['Algorithm'] = "HC"
        hc_df_long.loc[hc_df_long['Generation'] == 1, "Fitness"] = rs_df_long.loc[
            rs_df_long['Generation'] == 1, "Fitness"]

        df = pd.DataFrame(one_data[(program_name, qubit)])
        df['Run'] = df.index
        one_df_long = df.melt(id_vars='Run', var_name='Generation', value_name='Fitness')
        one_df_long['Generation'] = one_df_long['Generation'].astype(int) + 1
        one_df_long['Algorithm'] = "1+1"
        one_df_long.loc[one_df_long['Generation'] == 1, "Fitness"] = rs_df_long.loc[
            rs_df_long['Generation'] == 1, "Fitness"]

        DATA = pd.concat([rs_df_long, ga_df_long, hc_df_long, one_df_long],ignore_index=True)
        DATA["Program"] = program_name
        DATA["Qubit"] = qubit
        merged.append(DATA.copy())
    merged = pd.concat(merged, ignore_index=True)

    df_group = merged.groupby(["Run","Generation",'Algorithm',"Qubit"])
    df_fit = df_group.mean(numeric_only=True)
    df_fit.reset_index(inplace=True)

    for uq in df_fit["Qubit"].unique():
        plt.figure(figsize=(10, 6))
        pallet = sns.cubehelix_palette()
        plt.style.use('bmh')
        sns.pointplot(data=df_fit.loc[df_fit["Qubit"] == uq], x='Generation', y='Fitness', errorbar='sd', hue="Algorithm", estimator='mean',
                      palette=pallet)  # or ci=95 for 95% CI

        #plt.title('Best Fitness per Generation (10 Runs)')
        plt.title('')
        plt.xlabel('Generation', fontsize=18)
        plt.ylabel('Avg Best Fitness', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        # plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./graphs/generations_{uq}.png", bbox_inches='tight', dpi=300, transparent=True)
        #plt.show()


if __name__ == '__main__':
    #create_individual_graphs()
    #create_fitness_generation_graphs()
    create_fitness_generation_qubit_graphs()