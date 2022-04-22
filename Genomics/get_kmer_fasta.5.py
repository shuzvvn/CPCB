#!/usr/bin/env python3

# get_kmer_fasta.4.py

# Shu-Ting Cho <shutingc@andrew.cmu.edu>
# Read input fasta.gz
# calculate frequency of each kmer
# correct abundance with background (expected)
# output fasta files, each contain kmers that have the same corrected abundance
# report kmer seq in the output table
# report the positions in the genome of the kmers, and the differences between each position to see if it's tandem

# requires:  biopython
# v1 2022/04/21

# Usage:
# python3 get_kmer_fasta.1.py --in_file=GCF_003666465.1_ASM366646v1_genomic.fna.gz --out_dir=15955/ --k=12 --permute=20


## import modules
import time
print(time.strftime("%H:%M:%S", time.localtime()), '##### begin #####')

import argparse
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
#import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import itertools
import random
import re


# parsing arguments
def parse_args():
	parser = argparse.ArgumentParser(description='Calculate kmer frequencies of a given genome and generate kmer fasta files.')
	parser.add_argument('-i', '--in_file', help='input fasta.gz')
	parser.add_argument('-o', '--out_dir', help='output dir')
	parser.add_argument('-k', '--k', default="12", help='length of kmer sequence need to be calculated (default: 12)')
	parser.add_argument('-n', '--n_permute', default="1", help='number of permuted genome for background correction (default: 1)')
	parser.add_argument('-c', '--cutoff', default="30", help='cutoff count for reporting (default: 30)')
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

# find position of kmer
def kmer_position(kmer_h, seq_str):
	forward_pos = [m.start(0) for m in re.finditer(kmer_h, seq_str)]
	forward_pos_str = ', '.join([str(i) for i in forward_pos])
	diff_forward = (np.array(forward_pos[1:]) - np.array(forward_pos[:-1])).tolist()
	diff_forward_str = ', '.join([str(i) for i in diff_forward])

	rc_kmer_h = str(Seq(kmer_h).reverse_complement())
	reverse_pos = [m.start(0) for m in re.finditer(rc_kmer_h, seq_str)]
	reverse_pos_str = ', '.join([str(i) for i in reverse_pos])
	diff_reverse = (np.array(reverse_pos[1:]) - np.array(reverse_pos[:-1])).tolist()
	diff_reverse_str = ', '.join([str(i) for i in diff_reverse])

	return forward_pos_str, reverse_pos_str, diff_forward_str, diff_reverse_str


# main
def main():
	args = parse_args()

	in_file = args.in_file
	k = int(args.k)
	out_dir = args.out_dir.rstrip('/')
	n_permute = int(args.n_permute)
	cutoff = int(args.cutoff)

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
	print('cutoff = ', cutoff)
	
	kmer_mt_init_o = np.zeros(shape=[4]*k)
	kmer_mt_init_e = np.zeros(shape=[4]*k)
	seq_str_cat = ''

	## first contig
	# observed
	message = "start contig " + str(records[0].id)
	print(time.strftime("%H:%M:%S", time.localtime()), message)
	seq_str_f = str(records[0].seq.upper())
	seq_str_cat = seq_str_cat + seq_str_f
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
		seq_str_cat = seq_str_cat + seq_str_f
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


	# convert multidimensional matrix to a sorted kmer table
	print(time.strftime("%H:%M:%S", time.localtime()), 'convert multidimensional matrix to sorted kmer table')
	data = {'coord':list(itertools.product([0,1,2,3], repeat=k)), 'observed':kmer_mt_o.flatten(), 'expected':kmer_mt_e.flatten()}
	kmer_tab = pd.DataFrame(data)
	kmer_tab = kmer_tab[kmer_tab['observed']>1]
	kmer_tab['corrected'] = kmer_tab['observed'] - kmer_tab['expected'].round(0)
	kmer_tab = kmer_tab.sort_values('corrected', ignore_index=True, ascending=False)

	
	# saving as tsv file
	out_table_name = out_dir + '/kmer_table.tsv'

	# write kmers with same corrected frequency to fasta files
	counts = kmer_tab['corrected'].unique()
	counts = counts[counts>1]
	# remove not repeated kmers
	with open(out_table_name, 'w') as output_table_h:
		# write header to the output tsv
		output_table_h.write("Index\tKmer_seq\tobserved\texpected\tcorrected\tpos_f\tpos_r\tperiod_f\tperiod_r\n")
		for count_h in counts:
			df_h = kmer_tab[kmer_tab['corrected']==count_h]
			out_fasta_name = out_dir + '/kmers_' + str(int(count_h)) + '.fasta'
			print(time.strftime("%H:%M:%S", time.localtime()), 'writing output fasta:', out_fasta_name)
			with open(out_fasta_name, "w") as out_fasta_h:
				for i in df_h.index:
					coord_h = df_h.loc[i, 'coord']
					kmer_h = ''.join([str(value) for value in coord_h])
					kmer_h = kmer_h.replace('0', 'A')
					kmer_h = kmer_h.replace('1', 'T')
					kmer_h = kmer_h.replace('2', 'C')
					kmer_h = kmer_h.replace('3', 'G')
					my_SC = SeqRecord(Seq(kmer_h), str(i)+"_"+str(count_h), '', '')
					SeqIO.write(my_SC, out_fasta_h, "fasta")
					# table
					if count_h >= cutoff:
						pos_f, pos_r, diff_f, diff_r = kmer_position(kmer_h, seq_str_cat)
						output_table_h.write("%s\t%s\t%i\t%f\t%i\t%s\t%s\t%s\t%s\n" %(str(i), kmer_h, int(df_h.loc[i, 'observed']), df_h.loc[i, 'expected'], int(df_h.loc[i, 'corrected']), pos_f, pos_r, diff_f, diff_r))
					else:
						output_table_h.write("%s\t%s\t%i\t%f\t%i\t%s\t%s\t%s\t%s\n" %(str(i), kmer_h, int(df_h.loc[i, 'observed']), df_h.loc[i, 'expected'], int(df_h.loc[i, 'corrected']), '', '', '', ''))

	# plot
	print(time.strftime("%H:%M:%S", time.localtime()), 'plot rank_abundance.png')
	plot_file_name = out_dir + '/rank_abundance.png'
	kmer_tab = kmer_tab[kmer_tab['corrected']>10]
	kmer_tab = kmer_tab[['corrected', 'observed', 'expected']]
	kmer_tab = kmer_tab.reset_index(drop=True)
	kmer_tab['x'] = kmer_tab.index
	ax1 = kmer_tab.plot(kind='scatter', x='x', y='observed', color='b', s=10, label='observed')    
	ax2 = kmer_tab.plot(kind='scatter', x='x', y='expected', color='r', s=10, ax=ax1, label='expected')    
	ax3 = kmer_tab.plot(kind='scatter', x='x', y='corrected', color='g', s=10, ax=ax1, label='corrected')
	plt.xlabel('rank')
	plt.ylabel('count')
	plt.savefig(plot_file_name)
	print(time.strftime("%H:%M:%S", time.localtime()), '##### complete #####')

if __name__ == '__main__':
	main()
