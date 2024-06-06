"""
Plot the csv files produced by the evaluation. This plots the average recall, precision and f1 score against
the percentage of images considered "retrieved" (i.e. the top n images in the ranking). This is useful to see how
the retrieval performance changes as more images are considered, and as parameters between evaluations change (e.g. number of moments).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # Load the csv files
    files = [f for f in os.listdir("csv") if f.endswith(".csv")]
    files.sort()
    dfs = [pd.read_csv(f"csv/{f}") for f in files]

    # Plot the recall, precision and f1 score against the percentage of images considered
    ax, fig = plt.subplots(3, 1)
    for df in dfs:
        df = df.groupby("percentage").mean(numeric_only=True).reset_index()
        df = df.sort_values(by="percentage")

        fig[0].set_xlim([0, 0.2])
        fig[1].set_xlim([0, 0.2])
        fig[2].set_xlim([0, 0.2])

        fig[0].set_ylim([0.25, 1])
        fig[1].set_ylim([0, 1])
        fig[2].set_ylim([0, 1])

        fig[0].plot(df["percentage"], df["recall"])
        fig[0].set_title("Recall")

        fig[1].plot(df["percentage"], df["precision"])
        fig[1].set_title("Precision")

        fig[2].plot(df["percentage"], df["f1"])
        fig[2].set_title("F1 Score")

    # plt.legend([f"{f}" for f in files])
    plt.legend([f"{f[:-4]}" for f in files], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
