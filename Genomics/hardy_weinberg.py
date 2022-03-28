import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
from sklearn.preprocessing import StandardScaler

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

def compute_effective_allele_frequencies(split_snps):
    """Compute reference and alternate allele frequencies for each SNP.

    Arguments:
        split_snps: a (2 x num_snps x num_samples)-dimension Numpy array of strings "0" or "1"s, where snps[0] is the
            maternal haplotype for all samples and snps[1] is the paternal haplotype for all samples
    
    Returns:
        a 2-tuple of (snp_ref_frequency, snp_alt_frequency) representing the reference and alternate frequencies
            for each snp respectively, where snp_ref_frequency[i] + snp_alt_frequency[i] = 1
    """
    # for snp_ref_frequency
    # c11 is the number of "0" | "0" genotypes observed
    # c12 is the number of "0" | "1" genotypes observed
    # c21 is the number of "1" | "0" genotypes observed
    # c00 is the number of "1" | "1" genotypes observed
    # requency of the reference allele is 2c11+(c21+c12) / 2(c11+c21+c12+c22)
    
    num_snps, num_samples = split_snps.shape[1], split_snps.shape[2]
    g_00, g_01, g_10, g_11 = np.empty((num_snps)), np.empty((num_snps)), np.empty((num_snps)), np.empty((num_snps))

    for row in range(num_snps):
        g_00[row] = np.dot((split_snps[0,row]=='0').astype(int), (split_snps[1,row]=='0').astype(int))
        g_01[row] = np.dot((split_snps[0,row]=='0').astype(int), (split_snps[1,row]=='1').astype(int))
        g_10[row] = np.dot((split_snps[0,row]=='1').astype(int), (split_snps[1,row]=='0').astype(int))
        g_11[row] = np.dot((split_snps[0,row]=='1').astype(int), (split_snps[1,row]=='1').astype(int))

    snp_ref_frequency = (2*g_00+ (g_01+g_10)) / (2*(g_00+g_01+g_10+g_11))
    snp_alt_frequency = (2*g_11+ (g_01+g_10)) / (2*(g_00+g_01+g_10+g_11))

    return (snp_ref_frequency, snp_alt_frequency)
    pass

def compute_genotype_counts(split_snps):
    """Compute reference and alternate allele frequencies for each SNP.

    Arguments:
        split_snps: a (2 x num_snps x num_samples)-dimension Numpy array of strings "0" or "1"s, where snps[0] is the
            maternal haplotype for all samples and snps[1] is the paternal haplotype for all samples
    
    Returns:
        a 3-tuple of (homozygous_reference_counts, heterozygous_counts, homozygous_alternate_counts) representing the
            homozygous reference, heterozygous and homozygous alternate counts for each snp respectively,
            where snp_ref_frequency[i] + snp_alt_frequency[i] = 1
    """
    num_snps = split_snps.shape[1]
    g_00, g_01, g_10, g_11 = np.empty((num_snps)), np.empty((num_snps)), np.empty((num_snps)), np.empty((num_snps))

    for row in range(num_snps):
        g_00[row] = np.dot((split_snps[0,row]=='0').astype(int), (split_snps[1,row]=='0').astype(int))
        g_01[row] = np.dot((split_snps[0,row]=='0').astype(int), (split_snps[1,row]=='1').astype(int))
        g_10[row] = np.dot((split_snps[0,row]=='1').astype(int), (split_snps[1,row]=='0').astype(int))
        g_11[row] = np.dot((split_snps[0,row]=='1').astype(int), (split_snps[1,row]=='1').astype(int))

    homozygous_reference_counts = g_00
    heterozygous_counts = g_01+g_10
    homozygous_alternate_counts = g_11

    return (homozygous_reference_counts, heterozygous_counts, homozygous_alternate_counts)
    pass

def calculate_chi_squared_statistic(effective_allele_frequencies, genotype_counts):
    """Calculate the χ2 statistic for each SNP.

    Arguments:
        effective_allele_frequencies: a 2-tuple of (snp_ref_frequency, snp_alt_frequency) representing the reference and alternate frequencies
            for each snp respectively, where snp_ref_frequency[i] + snp_alt_frequency[i] = 1
        genotype_counts: a 3-tuple of (homozygous_reference_counts, heterozygous_counts, homozygous_alternate_counts)
            representing the homozygous reference, heterozygous and homozygous alternate counts for each snp respectively,
            where snp_ref_frequency[i] + snp_alt_frequency[i] = 1
    
    Returns:
        a num_snps-dimension Numpy array containing the χ2 statistic for each SNP.
                
    """
    num_samples = (genotype_counts[0] + genotype_counts[1] + genotype_counts[2])[0]
    homozygous_reference_counts, heterozygous_counts, homozygous_alternate_counts = genotype_counts[0], genotype_counts[1], genotype_counts[2]
    snp_ref_frequency, snp_alt_frequency = effective_allele_frequencies[0], effective_allele_frequencies[1]
    
    # Null Hypothesis: expected counts for each genotype based on the overall freq
    expected_homozygous_reference_counts = snp_ref_frequency**2 * num_samples
    expected_heterozygous_counts = snp_ref_frequency* snp_alt_frequency * 2 * num_samples
    expected_homozygous_alternate_counts = snp_alt_frequency**2 * num_samples

    # calculate chi2
    chi2_homozygous_reference = ((homozygous_reference_counts - expected_homozygous_reference_counts)**2)/expected_homozygous_reference_counts
    chi2_heterozygous = ((heterozygous_counts - expected_heterozygous_counts)**2)/expected_heterozygous_counts
    chi2_homozygous_alternate = ((homozygous_alternate_counts - expected_homozygous_alternate_counts)**2)/expected_homozygous_alternate_counts

    chi_squared_statistic = chi2_homozygous_reference + chi2_heterozygous + chi2_homozygous_alternate
    return chi_squared_statistic
    pass

def detect_snps_under_selection(chi_squared_statistic, alpha=1e-2, dof=1):
    """Determie which SNPs violate Hardy-Weinberg equilibrium according to the χ2 statistic.

    Arguments:
        a num_snps-dimension Numpy array containing the χ2 statistic for each SNP.
    
    Returns:
        a num_snps-dimension boolean Numpy array representing whether each SNP violates or doesn't violate HW-equilibrium
    """
    snps_under_selection = chi_squared_statistic > chi2.ppf(1-alpha, dof)
    return snps_under_selection
    pass

if __name__ == "__main__":
    phased_snps = pd.read_csv("../provided_data/chromosome_19_phased_snps.vcf", delimiter="\t")
    # TODO: make calls to above functions
    split_snps = split_phased_snps(phased_snps)
    effective_allele_frequencies = compute_effective_allele_frequencies(split_snps)
    genotype_counts = compute_genotype_counts(split_snps)
    chi_squared_statistic = calculate_chi_squared_statistic(effective_allele_frequencies, genotype_counts)
    snps_under_selection = detect_snps_under_selection(chi_squared_statistic, alpha=1e-2, dof=1)


