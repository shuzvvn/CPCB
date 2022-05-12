#!/usr/bin/env python3

from Bio import SeqIO
import sys

in_fasta = str(sys.argv[1])
region_list_file = str(sys.argv[2])
out_fasta = str(sys.argv[3])

record_dict = SeqIO.to_dict(SeqIO.parse(in_fasta, "fasta"))

with open(out_fasta, "w") as out_h:
	with open(region_list_file) as f_region:
		lines = f_region.readlines()
		for line in lines:
			words = line.rstrip('\n').split('\t')
			record_h = record_dict[words[0]]
			record_h = record_h[int(words[1])-1:int(words[2])]
			record_h.id = words[0] + '_' + words[1] + '_'+ words[2]
			record_h.name, record_h.description = '', ''
			SeqIO.write(record_h, out_h, "fasta")