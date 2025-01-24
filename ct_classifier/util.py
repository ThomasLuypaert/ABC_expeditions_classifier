"""
    Various utility functions used (possibly) across scripts.

    2022 Benjamin Kellenberger
"""

import random
import torch
from torch.backends import cudnn


def init_seed(seed):
    """
    Initalizes the seed for all random number generators used. This is
    important to be able to reproduce results and experiment with different
    random setups of the same code and experiments.
    """
    if seed is not None:
        random.seed(seed)
        # numpy.random.seed(seed)       # we don't use NumPy in this code, but you would want to set its random number generator seed, too
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        cudnn.deterministic = True


# 2. Function for creating a confusion matrix

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from ct_classifier.inference import model_inference


def create_confusion_matrix(summary_df, config, mapping_dict):

    # 1. Provide the true and predicted labels as strings
    true_labels = summary_df["true_labels"].unique()
    pred_labels = summary_df["pred_labels_full"].unique()

    # 2. Ensure the labels are provided as strings

    labels_string = [mapping_dict[x] for x in range(len(mapping_dict))]

    # 3. Create a normalized confusion matrix

    cm = confusion_matrix(
        summary_df["true_labels_full"],
        summary_df["pred_labels_full"],
        labels=labels_string,
        normalize="true",
    )

    # 4. Make the plot with labels

    fig, ax = plt.subplots(figsize=(20, 20))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels_string,
        yticklabels=labels_string,
        linecolor="lightgrey",  # Add thin grey lines
        linewidths=0.5,
        cbar=False,
    )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Normalized Confusion Matrix", size=25)

    return fig


# 3. Function for plotting a annotation-by-mistakes plot

import pandas as pd


def ann_by_mistake(summary_df):

    mismatches_df = pd.DataFrame(
        {
            "true_label": summary_df["true_labels_full"],
            "mismatch": (
                summary_df["pred_labels_full"] != summary_df["true_labels_full"]
            ),
        }
    )

    mismatches_df = pd.merge(
        mismatches_df.groupby("true_label").sum().reset_index("true_label"),
        mismatches_df["true_label"].value_counts().reset_index("true_label"),
        on="true_label",
    )

    mismatches_df["percentage_mistakes"] = (
        mismatches_df["mismatch"] / mismatches_df["count"]
    ) * 100
    print(mismatches_df)
    print(len(mismatches_df))

    mismatches_df["percentage_mistakes"] = mismatches_df["percentage_mistakes"].replace(
        0, 1e-10
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter_plot = sns.scatterplot(
        data=mismatches_df,
        x="percentage_mistakes",
        y="count",
        hue="true_label",
        palette="tab20",  # Categorical palette
        edgecolor="k",
        s=100,  # Size of markers
    )

    scatter_plot.legend_.remove()

    for i, row in mismatches_df.iterrows():
        plt.annotate(
            text=row["true_label"],
            xy=(row["percentage_mistakes"], row["count"]),
            xytext=(5, 5),  # Offset label slightly
            textcoords="offset points",
            fontsize=10,
            color="black",
        )

    scatter_plot.set_title("Annotations vs Mistakes (Training)", fontsize=16)
    scatter_plot.set_xlabel("Percentage of Mistakes", fontsize=14)
    scatter_plot.set_ylabel("Number of Annotations", fontsize=14)
    scatter_plot.grid(True)

    plt.xscale("log")
    plt.yscale("log")

    # plt.tight_layout()

    return fig


# Hierarchical class dictionaries

SPECIES2INDEX = {
    0: "curassow",
    1: "agouti-like",
    2: "tinamou",
    3: "paca",
    4: "southern_tamandua",
    5: "spixs_guan",
    6: "trumpeter",
    7: "brocket_deer",
    8: "longnosed_armadillo",
    9: "coati",
    10: "peccary",
    11: "ocelot",
    12: "margay",
    13: "nonground_birds",
    14: "jaguarundi",
    15: "marsupial",
    16: "squirrel",
    17: "tayra",
    18: "small_rodent",
    19: "giant_anteater",
    20: "giant_armadillo",
    21: "nakedtailed_armadillo",
    22: "shorteared_dog",
    23: "puma",
    24: "bats",
    25: "crabeating_raccoon",
    26: "southamerican_tapir",
    27: "porcupine",
    28: "ground_dove",
    29: "jaguar",
}

SPECIES2CLASS = {
    "curassow": "aves",
    "tinamou": "aves",
    "spixs_guan": "aves",
    "trumpeter": "aves",
    "nonground_birds": "aves",
    "ground_dove": "aves",
    "agouti-like": "mammalia",
    "paca": "mammalia",
    "southern_tamandua": "mammalia",
    "brocket_deer": "mammalia",
    "longnosed_armadillo": "mammalia",
    "coati": "mammalia",
    "peccary": "mammalia",
    "margay": "mammalia",
    "squirrel": "mammalia",
    "tayra": "mammalia",
    "marsupial": "mammalia",
    "small_rodent": "mammalia",
    "giant_armadillo": "mammalia",
    "nakedtailed_armadillo": "mammalia",
    "shorteared_dog": "mammalia",
    "bats": "mammalia",
    "crabeating_raccoon": "mammalia",
    "ocelot": "mammalia",
    "southamerican_tapir": "mammalia",
    "giant_anteater": "mammalia",
    "porcupine": "mammalia",
    "puma": "mammalia",
    "jaguar": "mammalia",
    "jaguarundi": "mammalia",
}

SPECIES2ORDER = {
    "curassow": "craciformes",
    "tinamou": "tinamiformes",
    "spixs_guan": "craciformes",
    "trumpeter": "gruiformes",
    "nonground_birds": "other_birds",
    "ground_dove": "columbiformes",
    "agouti-like": "rodentia",
    "paca": "rodentia",
    "southern_tamandua": "pilosa",
    "brocket_deer": "artiodactyla",
    "longnosed_armadillo": "cingulata",
    "coati": "carnivora",
    "peccary": "artiodactyla",
    "margay": "carnivora",
    "squirrel": "rodentia",
    "tayra": "carnivora",
    "marsupial": "didelphimorphia",
    "small_rodent": "rodentia",
    "giant_armadillo": "cingulata",
    "nakedtailed_armadillo": "cingulata",
    "shorteared_dog": "carnivora",
    "bats": "chiroptera",
    "crabeating_raccoon": "carnivora",
    "ocelot": "carnivora",
    "southamerican_tapir": "perissodactyla",
    "giant_anteater": "pilosa",
    "porcupine": "rodentia",
    "puma": "carnivora",
    "jaguar": "carnivora",
    "jaguarundi": "carnivora",
}

SPECIES2FAMILY = {
    "curassow": "cracidae",
    "tinamou": "tinamidae",
    "spixs_guan": "cracidae",
    "trumpeter": "psophidae",
    "nonground_birds": "other_birds_2",
    "ground_dove": "columbidae",
    "agouti-like": "dasyproctidae",
    "paca": "cuniculidae",
    "southern_tamandua": "myrmecophagidae",
    "brocket_deer": "cervidae",
    "longnosed_armadillo": "dasypodidae",
    "coati": "procyonidae",
    "peccary": "tayassuidae",
    "margay": "felidae",
    "squirrel": "sciuridae",
    "tayra": "mustelidae",
    "marsupial": "didelphidae",
    "small_rodent": "other_rod",
    "giant_armadillo": "dasypodidae",
    "nakedtailed_armadillo": "dasypodidae",
    "shorteared_dog": "canidae",
    "bats": "other_bats",
    "crabeating_raccoon": "procyonidae",
    "ocelot": "felidae",
    "southamerican_tapir": "tapiridae",
    "giant_anteater": "myrmecophagidae",
    "porcupine": "erethizontidae",
    "puma": "felidae",
    "jaguar": "felidae",
    "jaguarundi": "felidae",
}
