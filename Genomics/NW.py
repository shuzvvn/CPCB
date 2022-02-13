# NW.py
# HW1, Computational Genomics, Spring 2022
# andrewid: shutingc

# WARNING: Do not change the file name; Autograder expects it.

import sys
import numpy as np

def ReadFASTA(filename):
	fp=open(filename, 'r')
	Sequences={}
	tmpname=""
	tmpseq=""
	for line in fp:
		if line[0]==">":
			if len(tmpseq)!=0:
				Sequences[tmpname]=tmpseq
			tmpname=line.strip().split()[0][1:]
			tmpseq=""
		else:
			tmpseq+=line.strip()
	Sequences[tmpname]=tmpseq
	fp.close()
	return Sequences

# You may define any helper functions for Needleman-Wunsch algorithm here
match = 1
mismatch = -2
gap = 1

def s(ch1, ch2):
	score = 0
	if ch1.upper() == ch2.upper():
		score = match
	else:
		score = mismatch
	return score

# Do not change this function signature
def needleman_wunsch(seq1, seq2):
	"""Find the global alignment for seq1 and seq2
	Returns: 3 items as so:
	the alignment score, alignment in seq1 (str), alignment in seq2 (str)
	"""
	# initiation
	# build the traceback matrix F
	len_seq1 = len(seq1)
	len_seq2 = len(seq2)
	F = np.zeros([len_seq1+1, len_seq2+1])
	F[0,0] = 0
	F[:,0] = -np.arange(len_seq1+1) * gap
	F[0,:] = -np.arange(len_seq2+1) * gap
	
	# update
	for i in range(1,len_seq1+1):
		for j in range(1,len_seq2+1):
			diag = F[i-1, j-1] + s(seq1[i-1], seq2[j-1])
			left = F[i-1, j] - gap
			up = F[i, j-1] - gap
			F[i,j] = max( diag, left, up )

	# finding the optimal alignment
	i, j = len_seq1, len_seq2
	score = int(F[i, j])
	align1 = ""
	align2 = ""
	while [i, j] != [0, 0]:
		# tie-breaking occurs in the following order: (diagonal, left, top)
		if F[i, j] == F[i-1, j-1] + s(seq1[i-1], seq2[j-1]) : # diagonal
			[i, j] = [i-1, j-1]
			align1 = seq1[i] + align1
			align2 = seq2[j] + align2
		elif F[i, j] == F[i-1, j] - gap: # left
			[i, j] = [i-1, j]
			align1 = seq1[i] + align1
			align2 = "-" + align2
		else: # top
			[i, j] = [i, j-1]
			align1 = "-" + align1
			align2 = seq2[j] + align2

	return score, align1, align2
	raise NotImplementedError

if __name__=="__main__":
	Sequences=ReadFASTA(sys.argv[1])
	assert len(Sequences.keys())==2, "fasta file contains more than 2 sequences."
	seq1=Sequences[list(Sequences.keys())[0]]
	seq2=Sequences[list(Sequences.keys())[1]]

	score, align1, align2 = needleman_wunsch(seq1, seq2)

	print('Score: ', score)
	print('Seq1: ', align1)
	print('Seq2: ', align2)
