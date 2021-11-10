#!/usr/bin/python3

# hw1.5.py

# Shu-Ting Cho <shutingc@andrew.cmu.edu>
# In this problem, you will use census data that contains both categorical and continuous covariates to predict whether someone's income is >50K or <=50K. To do this you will be implementing KNN and Naive Bayes and comparing the difference between imputing and not imputing missing values.
# post-imputation

# v1 2021/09/22

# import modules
import numpy as np
import pandas as pd
from collections import Counter
import csv
import statistics
import math
import scipy.stats



# Loading CSV Files
df_train2 = pd.read_csv("df_train2.csv")
df_test2 = pd.read_csv("df_test2.csv")

# seperate data set into y=">50K" and y="<=50K"
df_train2_poor, df_train2_rich = [x for _, x in df_train2.groupby(df_train2['income'] == " >50K")]
print(df_train2_poor.shape, df_train2_rich.shape)

# 5.1 Report Parameters
# For questions below, report only up to 4 significant digits after the decimal points. In addition, for the questions below in this section use the data set with imputation of missing values.
# 1. [2 pts] Report the prior probability of each class.
print(20*"=", 5.1, 20*"=")
dict_df_train = {">50K": df_train2_rich, "<=50K": df_train2_poor}
dict_pY = {}
nrow_train = df_train2.shape[0]

for df in dict_df_train:
	n_Y = dict_df_train[df].shape[0]
	dict_pY[df] = n_Y / nrow_train
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
		print("\n#", feature_h, ":")
		print("Mean = %.4f, Variance = %.4f" % (mean_h, var_h))

	# If the attribute is discrete, report the value of P(xi=j|Y=c) for every possible value j in the boxes provided below!
	for feature_h in list_cate:
		print("\n#", feature_h, ":")
		dict_count = dict(Counter(dict_df_train[df][feature_h]))
		dict_N_par[df][feature_h] = {}
		for j in dict_count:
			p_j = dict_count[j]/nrow_df
			print("%s=%.4f" % (j , p_j))
			dict_N_par[df][feature_h][j] = p_j
	print("\n")

"""
# P(y) pi P(xi|y)
# clone test data to add posterior and prediction
df_test3 = df_test2
n_correct = 0
n = 0
for row_h in df_test3.index.tolist():
	n += 1
	posteriors = []
	for df in dict_df_train: # >50K, <=50K
		posterior_log_pyX = 0 # sum of log(P(xi|y)) + log(P(y))
		nrow_df = dict_df_train[df].shape[0]
		for col_h in dict_df_train[df].columns[:-2]: # features
			# calculate P(xi|y)
			value_h = df_test3.at[row_h, col_h]
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
		df_test3.at[row_h,df] = posterior_log_pyX
		posteriors.append(posterior_log_pyX)
	if posteriors[0] > posteriors[1]: # posteriors(">50K") > posteriors("<=50K")
		prediction = ">50K"
	else:
		prediction = "<=50K"
	df_test3.at[row_h,"prediction"] = prediction
	if prediction == df_test3.at[row_h, "income"].strip():
		n_correct += 1

print("accuracy:", n_correct/n)


df_test3.to_csv(r'df_test3.csv', index = False)

######################################################################################################
# Loading CSV Files
df_train = pd.read_csv("census.csv")

df_train_na = df_train

n_correct = 0
n = 0
for row_h in df_train_na.index.tolist():
	n += 1
	for df in dict_df_train: # >50K, <=50K
		posterior_log_pyX = 0 # sum of log(P(xi|y)) + log(P(y))
		nrow_df = dict_df_train[df].shape[0]
		for col_h in dict_df_train[df].columns[:-2]: # features
			# calculate P(xi|y)
			value_h = df_train_na.at[row_h, col_h]
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
		df_train_na.at[row_h,df] = posterior_log_pyX
		posteriors.append(posterior_log_pyX)
	if posteriors[0] > posteriors[1]: # posteriors(">50K") > posteriors("<=50K")
		prediction = ">50K"
	else:
		prediction = "<=50K"
	df_train_na.at[row_h,"prediction"] = prediction
	if prediction == df_train_na.at[row_h, "income"].strip():
		n_correct += 1
	# else:
	# 	print(prediction, df_train_na.at[row_h, "income"])

print("accuracy:", n_correct/n)

"""


######################################################################################################
# Loading CSV Files
df_train = pd.read_csv("census.csv")
df_test = pd.read_csv("adult.test.csv")
list_df_eva = [df_train, df_train2, df_test, df_test2]

for df_h in list_df_eva:

	n_correct = 0
	n = 0
	for row_h in df_h.index.tolist():
		n += 1
		posteriors = []
		for df in dict_df_train: # >50K, <=50K
			posterior_log_pyX = 0 # sum of log(P(xi|y)) + log(P(y))
			nrow_df = dict_df_train[df].shape[0]
			for col_h in dict_df_train[df].columns[:-2]: # features
				# calculate P(xi|y)
				value_h = df_h.at[row_h, col_h]
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
			df_h.at[row_h,df] = posterior_log_pyX
			posteriors.append(posterior_log_pyX)
		if posteriors[0] > posteriors[1]: # posteriors(">50K") > posteriors("<=50K")
			prediction = ">50K"
		else:
			prediction = "<=50K"
		df_h.at[row_h,"prediction"] = prediction
		if prediction == df_h.at[row_h, "income"].strip():
			n_correct += 1

	print("accuracy:", n_correct/n)