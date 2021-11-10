#!/usr/bin/python3

# ML_hw2.1.py

# Shu-Ting Cho <shutingc@andrew.cmu.edu>
# Data preprocessing

# v1 2021/09/29

import math

for A in list(range(0,50)):

	n_Total = 34+A

	n_long = 20
	n_short = 14+A

	n_like = 16+A
	n_Dlike = 18

	n_act = 3+5+4+8
	n_Dact = 9+3+A+2


	######################################################################3

	P_like = n_like / n_Total
	P_Dlike = n_Dlike / n_Total

	P_long = n_long / n_Total
	P_short = n_short / n_Total

	P_act = n_act / n_Total
	P_Dact = n_Dact / n_Total




	P_like_long = 12 / n_long
	P_Dlike_long = 8 / n_long

	P_like_short = (4+A) / n_short
	P_Dlike_short = 10 / n_short

	P_like_act = (3+4) / n_act
	P_Dlike_act = (5+8) / n_act

	P_like_Dact = (9+A) / n_Dact
	P_Dlike_Dact = (3+2) / n_Dact

	############################################################################################


	H_like = - (P_like * math.log(P_like,2)) - (P_Dlike * math.log(P_Dlike,2))

	#############################################################################################

	H_like_long = - (P_like_long * math.log(P_like_long,2)) - (P_Dlike_long * math.log(P_Dlike_long,2))

	H_like_short = - (P_like_short * math.log(P_like_short,2)) - (P_Dlike_short * math.log(P_Dlike_short,2))



	IG_len = H_like - ( (P_long*H_like_long) + (P_short*H_like_short) )

	############################################################################################

	H_like_act = - (P_like_act * math.log(P_like_act,2)) - (P_Dlike_act * math.log(P_Dlike_act,2))

	H_like_Dact = - (P_like_Dact * math.log(P_like_Dact,2)) - (P_Dlike_Dact * math.log(P_Dlike_Dact,2))

	IG_act = H_like - ( (P_act*H_like_act) + (P_Dact*H_like_Dact) )


	print("%i\t%.4f\t%.4f" % (A, IG_len, IG_act))