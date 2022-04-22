#!/Users/stc/opt/anaconda3/bin/python

# get_kmer_fasta.1.py

# Shu-Ting Cho <shutingc@andrew.cmu.edu>
# Read input fasta.gz
# calculate frequency of each kmer
# correct abundance with background (expected)
# output fasta files, each contain kmers that have the same corrected abundance

# requires:  biopython
# v1 2022/04/20

# Usage:
# python3 get_kmer_fasta.1.py --in_file=GCF_003666465.1_ASM366646v1_genomic.fna.gz --out_dir=15955/ --k=12 --permute=20


## import modules
import time
import argparse
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import itertools
import random


# parsing arguments
def parse_args():
	parser = argparse.ArgumentParser(description='Calculate kmer frequencies of a given genome and generate kmer fasta files.')
	parser.add_argument('-i', '--in_file', help='input fasta.gz')
	parser.add_argument('-o', '--out_dir', help='output dir')
	parser.add_argument('-k', '--k', default="12", help='length of kmer sequence need to be calculated (default: 12)')
	parser.add_argument('-p', '--n_permute', default="1", help='number of permuted genome for background correction (default: 1)')
	return parser.parse_args()


# The multidimensional method
def get_kmer_mt(seq_str, k=12, kmer_mt=np.zeros(shape=[4]*12)):
	seq_str = seq_str.replace('A', '0')
	seq_str = seq_str.replace('T', '1')
	seq_str = seq_str.replace('C', '2')
	seq_str = seq_str.replace('G', '3')
	i = 0
	while i+k <= len(seq_str):
		kmer_h = seq_str[i:i+k]
		coord = tuple([int(n) for n in kmer_h])
		kmer_mt[coord]+=1
		i+=1
	return kmer_mt

# shuffle sequence
def shuffle_seq(seq_str):
	permuted_seq_str = list(seq_str) 
	random.shuffle(permuted_seq_str)
	permuted_seq_str = ''.join(permuted_seq_str)
	return permuted_seq_str



# main
def main():
	args = parse_args()

	in_file = args.in_file
	k = int(args.k)
	out_dir = args.out_dir.rstrip('/')
	n_permute = int(args.n_permute)

	# start time
	print(time.strftime("%H:%M:%S", time.localtime()), "reading input")

	# make dir if not exist
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# read input
	with gzip.open(in_file, "rt") as handle:
		records = list(SeqIO.parse(handle, "fasta"))
	print('genome: ' + in_file)
	print('k =', k)
	print('n_permute = ', n_permute)
	
	kmer_mt_init_o = np.zeros(shape=[4]*k)
	kmer_mt_init_e = np.zeros(shape=[4]*k)

	## first contig
	# observed
	message = "start contig " + str(records[0].id)
	print(time.strftime("%H:%M:%S", time.localtime()), message)
	seq_str_f = str(records[0].seq.upper())
	print(time.strftime("%H:%M:%S", time.localtime()), 'forward strand (observed)')
	kmer_mt_o = get_kmer_mt(seq_str_f, k=k, kmer_mt=kmer_mt_init_o) # + strand
	seq_str_r = str(records[0].seq.reverse_complement().upper())
	print(time.strftime("%H:%M:%S", time.localtime()), 'reverse strand (observed)')
	kmer_mt_o = get_kmer_mt(seq_str_r, k=k, kmer_mt=kmer_mt_o) # - strand


	# expected (based on n permuted genomes)
	permuted_seq_f = shuffle_seq(seq_str_f)
	print(time.strftime("%H:%M:%S", time.localtime()), 'forward strand (expected, permutation 1 )')
	kmer_mt_e = get_kmer_mt(permuted_seq_f, k=k, kmer_mt=kmer_mt_init_e)
	permuted_seq_r = shuffle_seq(seq_str_r)
	print(time.strftime("%H:%M:%S", time.localtime()), 'reverse strand (expected, permutation 1 )')
	kmer_mt_e = get_kmer_mt(permuted_seq_r, k=k, kmer_mt=kmer_mt_e)
	for i in range(n_permute-1):
		permuted_seq_f = shuffle_seq(seq_str_f)
		print(time.strftime("%H:%M:%S", time.localtime()), 'forward strand (expected, permutation', i+2, ')')
		kmer_mt_e = get_kmer_mt(permuted_seq_f, k=k, kmer_mt=kmer_mt_e)
		permuted_seq_r = shuffle_seq(seq_str_r)
		print(time.strftime("%H:%M:%S", time.localtime()), 'reverse strand (expected, permutation', i+2, ')')
		kmer_mt_e = get_kmer_mt(permuted_seq_r, k=k, kmer_mt=kmer_mt_e)
		

	# loop through rest of the contigs
	for contig in range(1, len(records)):
		# observed
		message = "start contig " + str(records[contig].id)
		print(time.strftime("%H:%M:%S", time.localtime()), message)
		seq_str_f = str(records[contig].seq.upper())
		print(time.strftime("%H:%M:%S", time.localtime()), 'forward strand (observed)')
		kmer_mt_o = get_kmer_mt(seq_str_f, k=k, kmer_mt=kmer_mt_o) # + strand
		seq_str_r = str(records[contig].seq.reverse_complement().upper())
		print(time.strftime("%H:%M:%S", time.localtime()), 'reverse strand (observed)')
		kmer_mt_o = get_kmer_mt(seq_str_r, k=k, kmer_mt=kmer_mt_o) # - strand
		
		# expected
		for i in range(n_permute):
			permuted_seq_f = shuffle_seq(seq_str_f)
			print(time.strftime("%H:%M:%S", time.localtime()), 'forward strand (expected, permutation', i+1, ')')
			kmer_mt_e = get_kmer_mt(permuted_seq_f, k=k, kmer_mt=kmer_mt_e) # + strand
			permuted_seq_r = shuffle_seq(seq_str_r)
			print(time.strftime("%H:%M:%S", time.localtime()), 'reverse strand (expected, permutation', i+1, ')')
			kmer_mt_e = get_kmer_mt(permuted_seq_r, k=k, kmer_mt=kmer_mt_e) # - strand

	# mean for n permutation
	kmer_mt_e = kmer_mt_e/n_permute


	# convert multidimensional matrix to sorted kmer table
	print(time.strftime("%H:%M:%S", time.localtime()), 'convert multidimensional matrix to sorted kmer table')
	data = {'coord':list(itertools.product([0,1,2,3], repeat=k)), 'observed':kmer_mt_o.flatten(), 'expected':kmer_mt_e.flatten()}
	kmer_tab = pd.DataFrame(data)
	kmer_tab['corrected'] = kmer_tab['observed'] - kmer_tab['expected'].round(0)
	kmer_tab = kmer_tab.sort_values('corrected', ignore_index=True, ascending=False)

	# plot
	print(time.strftime("%H:%M:%S", time.localtime()), 'plot rank_abundance.png')
	plot_file_name = out_dir + '/rank_abundance.png'
	kmer_tab1 = kmer_tab[kmer_tab['observed']>1]
	kmer_tab1 = kmer_tab1[['corrected', 'observed', 'expected']]
	kmer_tab1.plot(figsize=(10, 10)).get_figure().savefig(plot_file_name)
	# saving as tsv file
	out_filename = out_dir + '/kmer_table.tsv'
	kmer_tab1.to_csv(out_filename, sep="\t")

	# write kmers with same corrected frequency to fasta files
	counts = kmer_tab['corrected'].unique()
	counts = counts[counts>1]
	# remove not repeated kmers

	for count_h in counts:
		df_h = kmer_tab[kmer_tab['corrected']==count_h]
		out_filename = out_dir + '/kmers_' + str(int(count_h)) + '.fasta'
		print(time.strftime("%H:%M:%S", time.localtime()), 'writing output fasta:', out_filename)
		with open(out_filename, "w") as output_handle:
			for i in df_h.index:
				coord_h = df_h.loc[i, 'coord']
				kmer_h = ''.join([str(value) for value in coord_h])
				kmer_h = kmer_h.replace('0', 'A')
				kmer_h = kmer_h.replace('1', 'T')
				kmer_h = kmer_h.replace('2', 'C')
				kmer_h = kmer_h.replace('3', 'G')
				my_SC = SeqRecord(Seq(kmer_h), str(i)+"_"+str(count_h), '', '')
				SeqIO.write(my_SC, output_handle, "fasta")

if __name__ == '__main__':
	main()