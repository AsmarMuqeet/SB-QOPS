import ast
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)

#NOISE_MODELS = ["ibm_brisbane", "ibm_torino", "ibm_kingston"]
NOISE_MODELS = ["ZNE_ibm_brisbane","ZNE_ibm_torino","ZNE_ibm_kingston"]
QUBITS = ["5", "10"]
#save_prefix = "noise"
save_prefix = "zne"

pname = {"ae":"Ae", "dj":"Dj", "ghz":"Ghz", "graphstate":"Graph", "qnn":"Qnn", "random":"Random", "realamprandom":"Real", "su2random":"Su2", "twolocalrandom":"Twolocal", "wstate":"Wstate"}

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
            df["Program"] = df["Program"].apply(lambda program: pname[program.split("_")[0]])
            df["avg_fitness"] = df["avg_fitness"].apply(ast.literal_eval)
            df = df.explode("avg_fitness")
            df["avg_fitness"] = df["avg_fitness"].astype(float)
            frames.append(df[["model", "qubit", "mutant", "Program", "avg_fitness"]])

    if not frames:
        return pd.DataFrame(
            columns=["model", "Program", "qubit", "mutant", "avg_fitness"]
        )

    return pd.concat(frames, ignore_index=True)


def plot_avg_fitness_boxplots(df: pd.DataFrame) -> None:
    """Create faceted box plots of average fitness per program."""
    sns.set_theme(style="whitegrid")

    df = df.copy()
    df["qubit_model"] = df["model"] + " | Q" + df["qubit"].astype(str)

    program_order = sorted(df["Program"].unique())
    df["Program"] = pd.Categorical(df["Program"], categories=program_order, ordered=True)

    facet_order = sorted(df["qubit_model"].unique(),key=lambda x: int(x.split(" | Q")[-1]))

    g = sns.catplot(
        data=df,
        x="Program",
        y="avg_fitness",
        hue="mutant",
        col="qubit_model",
        col_wrap=2,
        kind="box",
        sharey=True,
        height=5,
        aspect=1.1,
        order=program_order,
        col_order=facet_order,
    )

    g.set_axis_labels("Program", "Fitness",fontsize="16")
    g.set_titles("{col_name}")

    tick_positions = range(len(program_order))
    for ax in g.axes.flatten():
        ax.xaxis.set_major_locator(FixedLocator(tick_positions))
        ax.set_xticklabels(program_order, rotation=45, ha="right",fontsize="12")

    g.add_legend(title="Mutant")
    g.fig.subplots_adjust(top=0.88)
    g.fig.suptitle("Fitness Distribution For 10 Runs")

    output_path = os.path.join(os.getcwd(), f"{save_prefix}_avg_fitness_boxplots.png")
    g.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def plot_individual_avg_fitness(df: pd.DataFrame) -> None:
    """Create and save individual plots for each noise model/qubit combination."""
    sns.set_theme(style="whitegrid")

    df = df.copy()

    if df.empty:
        return

    program_order = sorted(df["Program"].unique())
    df["Program"] = pd.Categorical(df["Program"], categories=program_order, ordered=True)

    for (model, qubit), subset in df.groupby(["model", "qubit"]):
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=subset,
            x="Program",
            y="avg_fitness",
            hue="mutant",
            order=program_order,
        )
        ax.set_title(f"{model} | Q{qubit}")
        ax.set_xlabel("Program")
        ax.set_ylabel("Fitness")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Mutant")
        plt.tight_layout()

        filename = f"{save_prefix}_avg_fitness_boxplot_{model}_Q{qubit}.png"
        plt.savefig(os.path.join(os.getcwd(), filename), dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    dataframe = read_files()
    print(dataframe.head())
    plot_avg_fitness_boxplots(dataframe)
    #plot_individual_avg_fitness(dataframe)
