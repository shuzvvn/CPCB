# normalization.py
# HW2, Computational Genomics, Spring 2022
# andrewid: shutingc

# WARNING: Do not change the file name; Autograder expects it.

import sys
import numpy as np
import matplotlib.pyplot as plt

PER_MILLION = 1/1000000
PER_KILOBASE = 1/1000

# Do not change this function signature
def rpkm(raw_counts, gene_lengths):
    """Find the normalized counts for raw_counts
    Returns: a matrix of same size as raw_counts
    """
    RPKM = 10**9 * (raw_counts.T/gene_lengths).T / np.sum(raw_counts, axis=0)
    return RPKM
    pass
    
# Do not change this function signature
def tpm(raw_counts, gene_lengths):
    """Find the normalized counts for raw_counts
    Returns: a matrix of same size as raw_counts
    """
    RPK = raw_counts.T/(gene_lengths * PER_KILOBASE)
    scaling_factor = np.sum(RPK, axis=1) * PER_MILLION
    TPM = RPK.T / scaling_factor
    return TPM
    pass
   
# define any helper function here    


# Do not change this function signature


def size_factor(raw_counts):
    """Find the normalized counts for raw_counts
    Returns: a matrix of same size as raw_counts
    """
    den_h = np.prod(raw_counts, axis=1)**(1/24)
    gene_h = raw_counts.T / den_h
    sj = np.median(gene_h, axis=1)
    k = raw_counts/ sj
    return k
    pass
    

if __name__=="__main__":
    raw_counts=np.loadtxt(sys.argv[1])
    gene_lengths=np.loadtxt(sys.argv[2])
    
    rpkm1=rpkm(raw_counts, gene_lengths)
    tpm1=tpm(raw_counts, gene_lengths)
    size_factor1=size_factor(raw_counts)

    # TODO: write plotting code here

    fig = plt.figure(figsize=(20, 5))

    # i) raw counts
    ax = plt.subplot(1,4,1)

    ax.boxplot(raw_counts)
    ax.set_yscale('log', basey=2)
    ax.set_title('log2 RPKM')

    # ii) RPKM
    ax = plt.subplot(1,4,2)

    ax.boxplot(rpkm1)
    ax.set_yscale('log', basey=2)
    ax.set_title('log2 RPKM')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Gene expression level')

    # iii) TPM
    ax = plt.subplot(1,4,3)

    ax.boxplot(tpm1)
    ax.set_yscale('log', basey=2)
    ax.set_title('log2 TPM')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Gene expression level')

    # iv) size factor
    ax = plt.subplot(1,4,4)

    ax.boxplot(size_factor1)
    ax.set_yscale('log', basey=2)
    ax.set_title('log2 TPM')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Gene expression level')

    # Note2: Save your output from iv) as a new file, size factor normalized counts.txt
    np.savetxt("size_factor_normalized_counts.txt", size_factor1)
    pass
