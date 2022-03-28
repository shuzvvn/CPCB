import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def split_phased_snps(phased_snps):
    """Split phased SNP data into sequential pair of maternal and paternal SNPs.

    Arguments:
        phased_snps (pd.DataFrame): a (num_snps x (9 + num_samples)) dataframe loaded from a VCF file, with first
            nine columns ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] and the
            following columns containing the sequenced genotypes, where each value is a string of the form

                                    "maternal_allele | paternal_allele"

            for example, an entry "0 | 1" indicates that the maternal chromosome contained the reference allele
            at that locus while the paternal chromosome contained the alternate allele.

    Returns:
        split_snps: a (2 x num_snps x num_samples)-dimension Numpy array of strings "0" or "1"s, where snps[0] is the
            maternal haplotype for all samples and snps[1] is the paternal haplotype for all samples
    """
    df = phased_snps.iloc[:, 9:]
    num_snps, num_samples = df.shape[0], df.shape[1]

    split_snps = np.empty((2, num_snps, num_samples), dtype=str)

    for col in range(num_samples):
        split_snps[0,:,col] = df.iloc[:,col].str.extract("(\d)\|")[0]
        split_snps[1,:,col] = df.iloc[:,col].str.extract("\|(\d)")[0]

    return split_snps
    pass

def compute_allele_frequencies(snps):
    """Compute reference and alternate allele frequencies for each SNP.

    Arguments:
        snps: a (num_snps x (2 * num_samples))-dimension Numpy array of strings "0" or "1"s, where the first num_samples
            columns are alleles from the maternal chromosome and the second num_samples columns are the corresponding
            alleles from the paternal chromosome.

    Returns:
        a 2-tuple of (snp_ref_frequency, snp_alt_frequency) representing the reference and alternate frequencies
            for each snp respectively, where snp_ref_frequency[i] + snp_alt_frequency[i] = 1
    """
    num_alleles = snps.shape[1]

    snp_ref_frequency = (num_alleles - np.sum(snps=='1', axis=1))/num_alleles
    snp_alt_frequency = np.sum(snps=='1', axis=1)/num_alleles

    return (snp_ref_frequency, snp_alt_frequency)
    pass

def compute_haplotype_frequencies(snps):
    """Compute haplotype frequencies for every pair of SNPs.

    Arguments:
        snps: a (num_snps x (2 * num_samples))-dimension Numpy array of strings "0" or "1"s, where the first num_samples
            columns are alleles from the maternal chromosome and the second num_samples columns are the corresponding
            alleles from the paternal chromosome.

    Returns:
        a 4-tuple of (p_11, p_12, p_21, p_22) representing the haplotype frequencies of ("0", "0"), ("0", "1"),
            ("1", "0") and ("1", "1") for each pair respectively, where each element is a (num_snps x num_snps) Numpy
            array, and where the sum of p_11, p_12, p_21, p_22 for each pair of snps is 1.
    """
    num_alleles = snps.shape[1]

    p_11 = np.dot((snps=='0').astype(int), (snps=='0').astype(int).T)/num_alleles
    p_12 = np.dot((snps=='0').astype(int), (snps=='1').astype(int).T)/num_alleles
    p_21 = np.dot((snps=='1').astype(int), (snps=='0').astype(int).T)/num_alleles
    p_22 = np.dot((snps=='1').astype(int), (snps=='1').astype(int).T)/num_alleles

    return (p_11, p_12, p_21, p_22)
    pass

def calculate_D(haplotype_frequencies):
    """Calculate the linkage disequilibrium D.

    Arguments:
        haplotype_frequencies: a 4-tuple of (p_11, p_12, p_21, p_22) representing the haplotype frequencies of ("0", "0"), ("0", "1"),
            ("1", "0") and ("1", "1") for each pair respectively, where each element is a (num_snps x num_snps) Numpy
            array, and where the sum of p_11, p_12, p_21, p_22 for each pair of snps is 1.

    Returns:
        a (num_snps x num_snps) Numpy array of the raw linkage disequilibrium estimates for each pair of snps. This ranges in value
            from -0.25 to 0.25.
    """
    # D = p11 · p00 − p10 · p01
    p_11, p_12, p_21, p_22 = haplotype_frequencies[0], haplotype_frequencies[1], haplotype_frequencies[2], haplotype_frequencies[3]

    D = p_22*p_11 - p_21*p_12
    return D
    pass

def calculate_D_prime(allele_frequencies, haplotype_frequencies):
    """Calculate the standardized linkage disquilibrium D'.

    Arguments:
        allele_frequencies: a 2-tuple of (snp_ref_frequency, snp_alt_frequency) representing the reference and alternate frequencies
            for each allele respectively, where snp_ref_frequency[i] + snp_alt_frequency[i] = 1
        haplotype_frequencies: a 4-tuple of (p_11, p_12, p_21, p_22) representing the haplotype frequencies of ("0", "0"), ("0", "1"),
            ("1", "0") and ("1", "1") for each pair respectively, where each element is a (num_snps x num_snps) Numpy
            array, and where the sum of p_11, p_12, p_21, p_22 for each pair of snps is 1.

    Returns:
        a (num_snps x num_snps) Numpy array of the standardized linkage disequilibrium estimates for each pair of snps. This
            ranges in value from 0 to 1.
    """
    snp_ref_frequency, snp_alt_frequency = np.reshape(allele_frequencies[0], (-1,1)), np.reshape(allele_frequencies[1], (-1,1))
    D = calculate_D(haplotype_frequencies)

    p1q2 = np.dot(snp_ref_frequency, snp_alt_frequency.T)
    p2q1 = np.dot(snp_alt_frequency, snp_ref_frequency.T)
    p1q1 = np.dot(snp_ref_frequency, snp_ref_frequency.T)
    p2q2 = np.dot(snp_alt_frequency, snp_alt_frequency.T)
    
    # D′ = |D/Dmax|
    # D > 0, Dmax = min(p1q2, p2q1)
    # D <= 0, max(−p1q1,−p2q2)
    Dmax = np.where(D > 0, np.where(p1q2 < p2q1, p1q2, p2q1), np.where(-p1q1 > -p2q2, -p1q1, -p2q2))
    D_prime = np.abs(D/Dmax)

    return D_prime
    pass

def calculate_r_squared(allele_frequencies, haplotype_frequencies):
    """Calculate the square of Pearson's correlation coefficient r^2.

    Arguments:
        allele_frequencies: a 2-tuple of (snp_ref_frequency, snp_alt_frequency) representing the reference and alternate frequencies
            for each allele respectively, where snp_ref_frequency[i] + snp_alt_frequency[i] = 1
        haplotype_frequencies: a 4-tuple of (p_11, p_12, p_21, p_22) representing the haplotype frequencies of ("0", "0"), ("0", "1"),
            ("1", "0") and ("1", "1") for each pair respectively, where each element is a (num_snps x num_snps) Numpy
            array, and where the sum of p_11, p_12, p_21, p_22 for each pair of snps is 1.

    Returns:
        a (num_snps x num_snps) Numpy array of the r^2 values for each pair of snps. This
            ranges in value from 0 to 1.
    """
    snp_ref_frequency, snp_alt_frequency = np.reshape(allele_frequencies[0], (-1,1)), np.reshape(allele_frequencies[1], (-1,1))
    D = calculate_D(haplotype_frequencies)

    p1q2 = snp_ref_frequency * snp_alt_frequency
    p1p2q1q2 = np.dot(p1q2, p1q2.T)
    # r(P,Q) ≈ D/ √p1p2q1q2
    r_squared = (D/np.sqrt(p1p2q1q2))**2

    return r_squared
    pass

if __name__ == "__main__":
    phased_snps = pd.read_csv("../provided_data/chromosome_1_phased_first_5000_snps.vcf", delimiter="\t")
    # TODO: make calls to above functions
    split_snps = split_phased_snps(phased_snps)
    snps = np.c_[split_snps[0], split_snps[1]]
    allele_frequencies = compute_allele_frequencies(snps)
    haplotype_frequencies = compute_haplotype_frequencies(snps)
    D = calculate_D(haplotype_frequencies)
    D_prime = calculate_D_prime(allele_frequencies, haplotype_frequencies)
    r_squared = calculate_r_squared(allele_frequencies, haplotype_frequencies)

