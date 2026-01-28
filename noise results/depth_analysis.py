import ast
import os
import random

import matplotlib
import numpy as np
from qiskit import generate_preset_pass_manager


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

pd.set_option("display.max_columns", None)

#NOISE_MODELS = ["ibm_brisbane", "ibm_torino", "ibm_kingston"]
NOISE_MODELS = ["ZNE_ibm_brisbane","ZNE_ibm_torino","ZNE_ibm_kingston"]
QUBITS = ["5", "10"]

def make(x):
    n = 10
    r = n-len(x)
    a = []
    a.extend(x)
    a.extend(random.choices(x,k=r))
    return a

def read_files() -> pd.DataFrame:
    """Load and normalize the raw CSV outputs into a tidy dataframe."""
    frames = []
    for noise_model in NOISE_MODELS:
        for qubit in QUBITS:
            df = pd.read_csv(
                f"IBM1+1_{noise_model}_result{qubit}.csv",
                usecols=["Program", "mutant", "avg_fitness"],
            )
            df["model"] = noise_model
            df["qubit"] = qubit
            df["Program"] = df["Program"].apply(lambda program: program.split("_")[0])
            df["avg_fitness"] = df["avg_fitness"].apply(ast.literal_eval)
            df["avg_fitness"] = df["avg_fitness"].apply(make)
            df["run"] = "[1,2,3,4,5,6,7,8,9,10]"
            df["run"] = df["run"].apply(ast.literal_eval)
            print(noise_model,qubit)
            df = df.explode(["avg_fitness","run"])
            df["avg_fitness"] = df["avg_fitness"].astype(float)
            df["run"] = df["run"].astype(int)
            frames.append(df[["model", "qubit", "mutant", "Program", "avg_fitness","run"]])

    if not frames:
        return pd.DataFrame(
            columns=["model", "Program", "qubit", "mutant", "avg_fitness","run"]
        )

    return pd.concat(frames, ignore_index=True)




def make_threshold_analysis(
    data: pd.DataFrame,
    thresholds,
    score_col: str = "avg_fitness",
    label_col: str = "ground_truth",
    group_cols=("Program",),
) -> pd.DataFrame:
    """
    For each combination of group_cols and for each threshold in `thresholds`,
    compute accuracy, precision, recall, FP, FN, TP, TN.

    Prediction rule:
        predicted_label = 1 if score_col >= threshold else 0
    """



    rows = []

    for group_vals, sub in data.groupby(list(group_cols)):
        y_true = sub[label_col].values
        scores = sub[score_col].values

        for thr in thresholds:
            y_pred = (scores >= thr).astype(int)

            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())

            total = tp + tn + fp + fn

            accuracy  = (tp + tn) / total if total > 0 else np.nan
            precision = tp / (tp + fp)   if (tp + fp) > 0 else np.nan
            recall    = tp / (tp + fn)   if (tp + fn) > 0 else np.nan

            rows.append({
                "Program": group_vals[0],
                #"mutant": group_vals[1],
                #"model": group_vals[2],   # noise model
                #"qubit": group_vals[2],   # qubit count
                #"run": group_vals[3],
                "threshold": thr,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "true_negatives": tn,
                "n_samples": total,
            })

    return pd.DataFrame(rows)

def make_overall_threshold_analysis(
    data: pd.DataFrame,
    thresholds,
    score_col: str = "avg_fitness",
    label_col: str = "ground_truth",
) -> pd.DataFrame:
    """
    For each combination of group_cols and for each threshold in `thresholds`,
    compute accuracy, precision, recall, FP, FN, TP, TN.

    Prediction rule:
        predicted_label = 1 if score_col >= threshold else 0
    """



    rows = []


    y_true = data[label_col].values
    scores = data[score_col].values

    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        total = tp + tn + fp + fn

        accuracy  = (tp + tn) / total if total > 0 else np.nan
        precision = tp / (tp + fp)   if (tp + fp) > 0 else np.nan
        recall    = tp / (tp + fn)   if (tp + fn) > 0 else np.nan

        #f1 = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        balanced_accuracy = (recall + specificity) / 2

        rows.append({
            "threshold": thr,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "f1_score": f1,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "true_negatives": tn,
            #"n_samples": total,
        })

    return pd.DataFrame(rows)

if __name__ == '__main__':
    df = read_files()
    df.to_csv("classification_analysis.csv", index=False)
    print(df.head(10))

    df = pd.read_csv("classification_analysis.csv")

    median_tbl = (
        df.groupby(by=["model", "Program", "mutant", "qubit"])["avg_fitness"]
        .median()
        .reset_index()
    )
    median_tbl["ground_truth"] = 0
    median_tbl.loc[median_tbl["mutant"] != "eq", "ground_truth"] = 1
    median_tbl.to_csv("median_analysis.csv", index=False)

    thresholds = np.linspace(0.1, df["avg_fitness"].max(), 30).round(2)



    analysis_table = make_threshold_analysis(median_tbl, thresholds).round(2)
    analysis_table_ov = make_overall_threshold_analysis(median_tbl, thresholds).round(2)


    analysis_table_ov.to_csv("ov_threshold_analysis_table.csv", index=False)
    analysis_table.to_csv("threshold_analysis_table.csv", index=False)

