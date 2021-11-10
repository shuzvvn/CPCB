#!/usr/bin/python3

# hw1.1.py

# Shu-Ting Cho <shutingc@andrew.cmu.edu>
# In this problem, you will use census data that contains both categorical and continuous covariates to predict whether someone's income is >50K or <=50K. To do this you will be implementing KNN and Naive Bayes and comparing the difference between imputing and not imputing missing values.

# imputation
# v1 2021/09/20

# import modules
import numpy as np
import pandas as pd
from collections import Counter
import csv
import statistics
import math
import scipy.stats

import time

# start time
print("start", time.strftime("%H:%M:%S", time.localtime()))

## First you will create new training and test data sets with imputed missing values. For this we will be using a variant of K-Nearest Neighbor (KNN) algorithm with k = 10. 
k = 10

# First, combine the training and test data, remove the label from each input sample and then divide the combined data into two sets. The first set contains all input samples (rows) with no missing values while the second contains all those with at least one missing value. 


# Loading CSV Files
df_train = pd.read_csv("census.csv")
df_test = pd.read_csv("adult.test.csv")

df_train = df_train.assign(dataset="train")
df_test = df_test.assign(dataset="test")

print(df_test.shape)

################## subset #################
# df_train = df_train.head(500)
# df_test = df_test.head(500)


# combine the training and test data
data_combine = df_train.append(df_test)
data_combine = data_combine.set_axis(list(range(1,len(data_combine)+1)))


# clone data_combine for imputation
data_combine2 = data_combine

# remove the label from each input sample
data_combine = data_combine.iloc[:, :-2]


# divide the combined data into two sets. The first set contains all input samples (rows) with no missing values while the second contains all those with at least one missing value. 
data_combine_na = data_combine[data_combine.isnull().any(axis=1)]

# Drop rows which contain any NaN values
data_combine_nona = data_combine.dropna()

################## imputation KNN ##################
# Next, for each input sample in the second (missing values) set you would find its nearest neighbors in the first set. For this, we will use a distance metric to quantify how similar the rows are to each other.

n = 0
for row_na in data_combine_na.index.tolist():
	n += 1
	#print(n)
	# When calculating this metric you will use only the attributes that are not missing.
	# get cols for missing values
	col_missing = []

	# create an Empty DataFrame to store distances
	df_dist = pd.DataFrame()
	for col_h in data_combine_na.columns:
		value_h = data_combine_na.at[row_na, col_h]

		if pd.isna(value_h):
			col_missing.append(col_h)
		else:
			try:
				value_h = float(value_h)
				dist = abs(data_combine_nona[col_h] - value_h)
				df_dist[col_h] = dist
			except:
				value_h = str(value_h)
				df_dist[col_h] = (data_combine_nona[col_h] != value_h) * 10
	
	# To calculate the final distance just sum up the continuous and categorical distances for a pair of rows. 
	df_dist['sum'] = df_dist. sum(axis=1)
	
	# Following this you can rank, for each row with missing values, all rows that do not have missing values based on their distance and chose the top K.

	# rownames for top K
	k_rownames = df_dist.sort_values('sum').head(k).index.tolist()

	# Once you have the KNN for each missing value row you would impute the missing data as follows: 
	for col_h in col_missing:
		k_values = data_combine_nona.loc[k_rownames, col_h].tolist()
		try:
			# For continuous values use the average of the values for this attribute in the KNN rows you identified. For categorical values use the majority value.
			final_value = statistics.mean(k_values)
		except:
			k_values_occurrences = Counter(k_values).most_common()
			# In case of ties for categorical values (i.e. two or more categorical values have the same top number of appearances in the KNN), break them based on the total number of times the values appear in the entire dataset, choosing the category with the most number of occurrences. 
			if len(k_values_occurrences) > 1 and k_values_occurrences[0][1] == k_values_occurrences[1][1]:
				times = k_values_occurrences[0][1]
				highest_occurrences = 0
				for i in k_values_occurrences:
					if i[1] == times:
						cand_occurrence_all = data_combine[col_h].tolist().count(i[0])
						if data_combine[col_h].tolist().count(i[0]) > highest_occurrences:
							highest_occurrences = cand_occurrence_all
							final_value = i[0]
			else:
				final_value = k_values_occurrences[0][0]
		# impute missing values to data_combine2
		data_combine2.at[row_na, col_h] = final_value
		output_list = data_combine2.loc[row_na,:].tolist()

print("completed", time.strftime("%H:%M:%S", time.localtime()))

## After imputation, you will implement the Naive Bayes (NB) algorithm on both the original training data set and the new training data set with imputed values.
# seperate train and test
df_train2, df_test2 = [x for _, x in data_combine2.groupby(data_combine2['dataset'] == "test")]
df_train2.to_csv(r'df_train2.csv', index = False)
df_test2.to_csv(r'df_test2.csv', index = False)


###################################################################################################


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