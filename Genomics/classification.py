# classification.py
# HW2, Computational Genomics, Spring 2022
# andrewid: shutingc

# WARNING: Do not change the file name; Autograder expects it.

import sys

import numpy as np
from scipy.sparse import csc_matrix, save_npz, load_npz

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def get_top_gene_filter(data, n_keep = 2000):
    """Select top n_keep most dispersed genes.

    Args:
        data (n x m matrix): input gene expression data of shape num_cells x num_genes
        n_keep (int): number of genes to be kepted after filtration; default 2000

    Returns:
        filter (array of length n_keep): an array of column indices that can be used as an
            index to keep only certain genes in data. Each element of filter is the column
            index of a highly-dispersed gene in data.
    """
    var_h = np.var(data, axis=0)
    mean_h = np.mean(data, axis=0)
    disp_h = var_h/mean_h
    index_filter = (-disp_h).argsort()[:n_keep]
    return index_filter
    pass

def reduce_dimensionality_pca(filtered_train_gene_expression, filtered_test_gene_expression, n_components = 20):
    """Train a PCA model and use it to reduce the training and testing data.
    
    Args:
        filtered_train_gene_expression (n_train x num_top_genes matrix): input filtered training expression data 
        filtered_test_gene_expression (n_test x num_top_genes matrix): input filtered test expression data 
        
    Return:
        (reduced_train_data, reduced_test_data): a tuple of
            1. The filtered training data transformed to the PC space.
            2. The filtered test data transformed to the PC space.
    """
    concat_data = np.concatenate((filtered_train_gene_expression, filtered_test_gene_expression), axis=0)
    scaler = StandardScaler()
    scaler.fit(concat_data)
    concat_data = scaler.transform(concat_data)
    filtered_train_gene_expression = scaler.transform(filtered_train_gene_expression)
    filtered_test_gene_expression = scaler.transform(filtered_test_gene_expression)

    pca = PCA(n_components=n_components)
    pca.fit(concat_data)
    reduced_train_data = pca.transform(filtered_train_gene_expression)
    reduced_test_data = pca.transform(filtered_test_gene_expression)
    return (reduced_train_data, reduced_test_data)   
    pass

def plot_transformed_cells(reduced_train_data, train_labels):
    """Plot the PCA-reduced training data using just the first 2 principal components.
    
    Args:
        reduced_train_data (n_train x num_components matrix): reduced training expression data
        train_labels (array of length n_train): array of cell type labels for training data
        
    Return:
        None

    """
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    targets = np.unique(train_labels)
    for target in targets:
        ax.scatter(reduced_train_data[train_labels==target,0]
                   , reduced_train_data[train_labels==target,1]
                   , s = 10)
    ax.legend(targets)
    ax.grid()
    pass
    
def train_and_evaluate_svm_classifier(reduced_train_data, reduced_test_data, train_labels, test_labels):
    """Train and evaluate a simple SVM-based classification pipeline.
    
    Before passing the data to the SVM module, this function scales the data such that the mean
    is 0 and the variance is 1.
    
    Args:
        reduced_train_data (n_train x num_components matrix): reduced training expression data
        train_labels (array of length n_train): array of cell type labels for training data
        
    Return:
        (classifier, score): a tuple consisting of
            1. classifier: the trained classifier
            2. The score (accuracy) of the classifier on the test data.

    """
    classifier = SVC()
    classifier.fit(reduced_train_data, train_labels)
    score_train = classifier.score(reduced_train_data, train_labels)
    score_test = classifier.score(reduced_test_data, test_labels)
    return (classifier, score_train, score_test)
    pass
        
if __name__ == "__main__":
    train_gene_expression = load_npz(sys.argv[1]).toarray()
    test_gene_expression = load_npz(sys.argv[2]).toarray()
    train_labels = np.load(sys.argv[3])
    test_labels = np.load(sys.argv[4])
    
    top_gene_filter = get_top_gene_filter(train_gene_expression)
    filtered_test_gene_expression = test_gene_expression[:, top_gene_filter]
    filtered_train_gene_expression = train_gene_expression[:, top_gene_filter]
        
    mode = sys.argv[5]
    if mode == "svm_pipeline":
        # TODO: Implement the pipeline here
        # PCA dimensionality reduction
        (reduced_train_data, reduced_test_data) = reduce_dimensionality_pca(filtered_train_gene_expression, filtered_test_gene_expression, n_components = 20)
        # SVM classification
        (classifier, score_train, score_test) = train_and_evaluate_svm_classifier(reduced_train_data, reduced_test_data, train_labels, test_labels)
        print("training accuracy:", score_train)
        print("test accuracy:", score_test)
        pass
