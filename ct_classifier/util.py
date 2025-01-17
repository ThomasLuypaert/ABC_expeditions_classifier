'''
    Various utility functions used (possibly) across scripts.

    2022 Benjamin Kellenberger
'''

import random
import torch
from torch.backends import cudnn


def init_seed(seed):
    '''
        Initalizes the seed for all random number generators used. This is
        important to be able to reproduce results and experiment with different
        random setups of the same code and experiments.
    '''
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

    #1. Provide the true and predicted labels as strings
    true_labels = df_summary["true_labels"].unique()
    pred_labels = df_summary["pred_labels_full"].unique()
    
    # 2. Ensure the labels are provided as strings
    
    labels_string = [mapping_dict[x] for x in range(len(mapping_dict))]

    # 3. Create a normalized confusion matrix
    
    cm = confusion_matrix(summary_df["true_labels_full"],
                          summary_df["pred_labels_full"],
                          labels=labels_string, 
                          normalize = "true")
    
    # 4. Make the plot with labels
    
    fig, ax = plt.subplots(figsize=(20, 20))
    
    sns.heatmap(cm,
                annot=True, 
                fmt='.2f', 
                  cmap="Blues", 
                  xticklabels = labels_string, 
                  yticklabels = labels_string, 
                  linecolor='lightgrey',  # Add thin grey lines
                    linewidths=0.5,
                    cbar=False)  
    
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Normalized Confusion Matrix',size = 25)

    return plt


# 3. Function for plotting a annotation-by-mistakes plot

import pandas as pd

def ann_by_mistake(summary_df):
    
    n_annotations = summary_df['true_labels_full'].value_counts()

    mismatches_df = pd.DataFrame({"true_label":summary_df['true_labels_full'],
                              "mismatch":(summary_df['pred_labels_full'] != summary_df['true_labels_full'])})
    
    mismatches_df = pd.merge(mismatches_df.groupby("true_label").sum().reset_index("true_label"), 
                        mismatches_df["true_label"].value_counts().reset_index("true_label"), 
                        on='true_label')
    
    mismatches_df["percentage_mistakes"] = (mismatches_df["mismatch"] / mismatches_df["count"])*100
    
    plt.figure(figsize=(12, 8))
    
    scatter_plot = sns.scatterplot(
    data=mismatches_df,
    x='percentage_mistakes',
    y='count',
    hue='true_label',
    palette='tab20',  # Categorical palette
    edgecolor='k',
    s=100  # Size of markers
    )
    
    scatter_plot.legend_.remove()
    
    for i, row in mismatches_df.iterrows():
        plt.annotate(
        text=row['true_label'],
        xy=(row['percentage_mistakes'], row['count']),
        xytext=(5, 5),  # Offset label slightly
        textcoords='offset points',
        fontsize=10,
        color='black'
    )
        
    scatter_plot.set_title("Annotations vs Mistakes (Training)", fontsize=16)
    scatter_plot.set_xlabel("Percentage of Mistakes", fontsize=14)
    scatter_plot.set_ylabel("Number of Annotations", fontsize=14)
    scatter_plot.grid(True)
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()

    return plt