# de_genes.py
# HW2, Computational Genomics, Spring 2022
# andrewid: shutingc

# WARNING: Do not change the file name; Autograder expects it.

import sys
import numpy as np


# Do not change this function signature

def bh(genes, pvals, alpha):
    """(list, list, float) -> numpy array
    applies benjamini-hochberg procedure
    
    Parameters
    ----------
    genes: name of genes 
    pvalues: corresponding pvals
    alpha: desired false discovery rate
    
    Returns
    -------
    array containing gene names of significant genes.
    gene names do not need to be in any specific order.
    """
    index_ordered_p = (np.array(pvals)).argsort()
    ordered_pvals = [pvals[i] for i in index_ordered_p]
    ordered_genes = [genes[i] for i in index_ordered_p]

    n = len(pvals)

    i = 1
    highest_i = 0
    while i <= n:
        if ordered_pvals[i-1] <= (i/n) * alpha:
            highest_i = i
        i+=1
    accept_genes = [ordered_genes[i] for i in range(highest_i)]
    return accept_genes
    pass

# define any helper function here    

if __name__=="__main__":
    # Here is a free test case
    genes=['a', 'b', 'c']
    input1 = [0.01, 0.04, 0.1]
    print(bh(genes, input1, 0.05))
