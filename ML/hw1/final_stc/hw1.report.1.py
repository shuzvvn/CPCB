#!/usr/bin/python3

# hw1.report.1.py

# Shu-Ting Cho <shutingc@andrew.cmu.edu>

# post-imputation
# For questions below, report only up to 4 significant digits after the decimal points. In addition, for the questions below in this section use the data set with imputation of missing values.
# v1 2021/09/22

# import modules
import pandas as pd
from collections import Counter
import statistics
import math


# Loading CSV Files (w/ imputation)
df_train2 = pd.read_csv("df_train2.csv")
df_test2 = pd.read_csv("df_test2.csv")

# seperate training data set into y=">50K" and y="<=50K"
df_train2_poor, df_train2_rich = [x for _, x in df_train2.groupby(df_train2['income'] == " >50K")]

############################################### 5.1 ###############################################

##### 5.1 Report Parameters #####
# For questions below, report only up to 4 significant digits after the decimal points. In addition, for the questions below in this section use the data set with imputation of missing values.
# 1. [2 pts] Report the prior probability of each class.
print(20*"=", "5.1", 20*"=")
dict_df_train = {">50K": df_train2_rich, "<=50K": df_train2_poor}
dict_pY = {}
nrow_train = df_train2.shape[0]

for df in dict_df_train:
	nrow_Y = dict_df_train[df].shape[0]
	dict_pY[df] = nrow_Y / nrow_train
	print("5.1.1 prior P(income%s): %.4f" % (df, dict_pY[df]))
print("\n")


# 2. For each class c and for each attribute i in [education-num, marital-status, race, capital-gain] print & report the following:
print(18*"=", "5.1.2", 18*"=")
list_cont = ["education-num", "capital-gain", "age", "capital-loss", "hours-per-week", "fnlwgt"]
list_cate = ["marital-status", "race", "workclass", "occupation", "education", "relationship", "sex", "native-country"]

dict_N_par = {}

# loop to report value for 5.1.2
for df in dict_df_train:
	dict_N_par[df] = {} # parameters for >50 or <=50
	print(16*"=", df, 16*"=")
	nrow_df = dict_df_train[df].shape[0]
	# If the attribute is continuous, report the value of mean and variance in their corresponding boxes.
	for feature_h in list_cont:
		mean_h = statistics.mean(dict_df_train[df][feature_h])
		var_h = statistics.pvariance(dict_df_train[df][feature_h])
		dict_N_par[df][feature_h] = [mean_h, var_h]
		# print for features asked by 5.1.2
		if feature_h in list_cont[:2]:
			print("\n#", feature_h, ":")
			print("Mean = %.4f, Variance = %.4f" % (mean_h, var_h))

	# If the attribute is discrete, report the value of P(xi=j|Y=c) for every possible value j in the boxes provided below!
	for feature_h in list_cate:
		dict_count = dict(Counter(dict_df_train[df][feature_h]))
		dict_N_par[df][feature_h] = {}
		for j in dict_count:
			p_j = dict_count[j]/nrow_df
			dict_N_par[df][feature_h][j] = p_j
		# print for features asked by 5.1.2
		if feature_h in list_cate[:2]:
			print("\n#", feature_h, ":")
			for i in dict_N_par[df][feature_h]:
				print("%s: %.4f" % (i.strip(), dict_N_par[df][feature_h][i]))
			#print("%s=%.4f" % (j , p_j))
	print("\n")


############################################### 5.2.1-4 ###############################################
# Loading CSV Files
df_train = pd.read_csv("census.csv")
df_test = pd.read_csv("adult.test.csv")
dict_df_eva = {"5.2.1":df_train, "5.2.2":df_train2, "5.2.3":df_test, "5.2.4":df_test2}


print(20*"=", "5.2", 20*"=")
for df_h in dict_df_eva:
	n_correct = 0
	n = 0
	for row_h in dict_df_eva[df_h].index.tolist():
		n += 1
		posteriors = []
		for df in dict_df_train: # >50K, <=50K
			posterior_log_pyX = 0 # sum of log(P(xi|y)) + log(P(y))
			nrow_df = dict_df_train[df].shape[0]
			for col_h in dict_df_train[df].columns[:-2]: # features
				# calculate P(xi|y)
				value_h = dict_df_eva[df_h].at[row_h, col_h]
				if not pd.isna(value_h):
					# cont or cate
					try: # cont
						value_h = float(value_h)
						# mean and var from dict_N_par[df][col_h]
						mean_h = dict_N_par[df][col_h][0]
						var_h = dict_N_par[df][col_h][1]+10**-9 # var + epsilon
						p_value_h = (1 / math.sqrt(2*math.pi*var_h)) * math.exp(-(((value_h-mean_h)**2)/(2*var_h)))
					except: # cate
						value_h = str(value_h)
						try:
							p_value_h = dict_N_par[df][col_h][value_h]
						except:
							p_value_h = 0
						# add log value to the sum
					if p_value_h != 0:
						p_value_h = math.log(p_value_h)
					else:
						p_value_h = -math.inf
					posterior_log_pyX = posterior_log_pyX + p_value_h
			posterior_log_pyX = math.log(dict_pY[df]) + posterior_log_pyX
			dict_df_eva[df_h].at[row_h,df] = posterior_log_pyX
			posteriors.append(posterior_log_pyX)
		if posteriors[0] < posteriors[1]: # posteriors(">50K") < posteriors("<=50K")
			prediction = "<=50K"
		else:
			prediction = ">50K"
		dict_df_eva[df_h].at[row_h,"prediction"] = prediction
		if prediction == dict_df_eva[df_h].at[row_h, "income"].strip():
			n_correct += 1

	print("%s: %.4f" % (df_h, n_correct/n))
print("\n")

################################################ 5.1.3 ################################################
print(18*"=", "5.1.3", 18*"=")
print("<=50K\t\t>50K")

first10test = dict_df_eva["5.2.4"].head(10)
for row_h in first10test.index:
	print("%.4f\t%.4f" % (first10test.at[row_h,"<=50K"], first10test.at[row_h,">50K"]))