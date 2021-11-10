#!/usr/bin/python3

# hw1.1.py

# Shu-Ting Cho <shutingc@andrew.cmu.edu>
# In this problem, you will use census data that contains both categorical and continuous covariates to predict whether someone's income is >50K or <=50K. To do this you will be implementing KNN and Naive Bayes and comparing the difference between imputing and not imputing missing values.

# requires:  numpy
# v1 2021/09/20

# import modules
import numpy as np
import pandas as pd
from collections import Counter
import csv
from statistics import mean


## First you will create new training and test data sets with imputed missing values. For this we will be using a variant of K-Nearest Neighbor (KNN) algorithm with k = 10. 
k = 10

# First, combine the training and test data, remove the label from each input sample and then divide the combined data into two sets. The first set contains all input samples (rows) with no missing values while the second contains all those with at least one missing value. 


# Loading CSV Files
df_train = pd.read_csv("census.csv")
df_test = pd.read_csv("adult.test.csv")

df_train = df_train.set_axis(list(range(1,len(df_train)+1)))



# combine the training and test data
data_combine = df_train.append(df_test)
data_combine = data_combine.set_axis(list(range(1,len(data_combine)+1)))
# print(df_train.shape, df_test.shape, data_combine.shape)
# print(data_combine.set_axis(list(range(1,len(data_combine)+1))))
# data_combine = np.vstack([data_train, data_test])
#print(data_train.shape, data_test.shape, data_combine.shape)


# remove the label from each input sample
#data_combine = data_combine[:,:-1]
data_combine = data_combine.iloc[:, :-1]
print(data_combine.shape)

# np.isnan(np.array([np.nan, 0], dtype=np.float64))
# print(~np.isnan(data_combine))


# divide the combined data into two sets. The first set contains all input samples (rows) with no missing values while the second contains all those with at least one missing value. 
data_combine_na = data_combine[data_combine.isnull().any(axis=1)]
#data_combine_na.to_numpy()

# Drop rows which contain any NaN values
data_combine_nona = data_combine.dropna()
#data_combine_nona.to_numpy()

print(data_combine_na.shape, data_combine_nona.shape)
# print(data_combine_na, data_combine_nona)
#print(data_combine_na.index).tolist()


################## subset #################
# data_combine_na = data_combine_na.head(10)
# data_combine_nona = data_combine_nona.head(200)

# clone data_combine_na for imputing
data_combine_na2 = data_combine_na



# for row in data_combine_na:
# 	for cell in row:
# 		print(cell)

# 	n += 1

## KNN
# Next, for each input sample in the second (missing values) set you would find its nearest neighbors in the first set. For this, we will use a distance metric to quantify how similar the rows are to each other.

n = 0

for row_na in data_combine_na.index.tolist():
	# When calculating this metric you will use only the attributes that are not missing.
	# get cols for missing values, continuous and categorical attributes
	col_missing = []
	#col_cont = []
	#col_cate = []
	n += 1
	print(n, "="*50)

	# create an Empty DataFrame to store distances
	df_dist = pd.DataFrame()
	#df_dist = data_combine_na
	df_dist["name"] = data_combine_nona.index
	#df_dist = df_dist.assign(sum=0)
	df_dist.index = data_combine_nona.index
	df_dist = df_dist.drop("name", axis=1)
	#print(df_dist)

	for col_h in data_combine_na.columns:
		value_h = data_combine_na.at[row_na, col_h]
		#print(col_h, value_h)
		if pd.isna(value_h):
			col_missing.append(col_h)
		else:
			try:
				value_h = float(value_h)
				# print(col_h, value_h, "cont")
				dist = abs(data_combine_nona[col_h] - value_h)
				df_dist[col_h] = dist
				# print(abs(data_combine_nona[col_h] - value_h))
				#print(df_dist[col_h])
				#col_cont.append(col_h)
			except:
				value_h = str(value_h)
				# print(col_h, value_h, "cate")
				#df_dist[col_h] = 0
				# print(data_combine_nona[col_h] != value_h)
				#list_com = (data_combine_nona[col_h] != value_h) * 10
				#print(list_com)
				df_dist[col_h] = (data_combine_nona[col_h] != value_h) * 10
				#print(df_dist[col_h])

				#print(data_combine_nona[col_h] == value_h)
				#col_cate.append(col_h)
	
	# To calculate the final distance just sum up the continuous and categorical distances for a pair of rows. 
	df_dist['sum'] = df_dist. sum(axis=1)
	
	# Following this you can rank, for each row with missing values, all rows that do not have missing values based on their distance and chose the top K.

	# rownames for top K
	#print(df_dist.sort_values('sum').head(k).index)
	k_rownames = df_dist.sort_values('sum').head(k).index.tolist()
	#print(k_rownames)
	for col_h in col_missing:
		k_values = data_combine_nona.loc[k_rownames, col_h].tolist()
		#print(k_values)
		try:
			final_value = mean(k_values)
		except:
			k_values_occurrences = Counter(k_values).most_common()
			if len(k_values_occurrences) > 1 and k_values_occurrences[0][1] == k_values_occurrences[1][1]:
				times = k_values_occurrences[0][1]
				cand = []
				highest_occurrences = 0
				for i in k_values_occurrences:
					if i[1] == times:
						cand_occurrence_all = data_combine[col_h].tolist().count(i[0])
						if cand_occurrence_all > highest_occurrences:
							final_value = i[0]
			else:
				final_value = k_values_occurrences[0][0]
			#print(col_h, final_value)

		data_combine_na2.at[row_na, col_h] = final_value
		#print(data_combine_na2.loc[[row_na]])

data_check_na = data_combine_na2[data_combine_na2.isnull().any(axis=1)]
print(data_check_na)




"""

# divide the combined data into two sets. The first set contains all input samples (rows) with no missing values while the second contains all those with at least one missing value. 
#print(np.isnan(data_combine))


# Next, for each input sample in the second (missing values) set you would find its nearest neighbors in the first set. For this, we will use a distance metric to quantify how similar the rows are to each other. When calculating this metric you will use only the attributes that are not missing. For continuous attributes you will use Euclidean distance. For categorical attributes the distance will be 0 if the two categories are the same and 10 if they are different. To calculate the final distance just sum up the continuous and categorical distances for a pair of rows. Following this you can rank, for each row with missing values, all rows that do not have missing values based on their distance and chose the top K. Once you have the KNN for each missing value row you would impute the missing data as follows: For continuous values use the average of the values for this attribute in the KNN rows you identified. For categorical values use the majority value.

# In case of ties for categorical values (i.e. two or more categorical values have the same top number of appearances in the KNN), break them based on the total number of times the values appear in the entire dataset, choosing the category with the most number of occurrences. 







## import modules


# get all combinations of pairs from list of seq IDs
def pairs(IDs):
	pairs_list = combinations(IDs, 2)
	return pairs_list

# get identity of two seqs
def identity(seq_pair):
	seq1 = seq_pair[0]
	seq2 = seq_pair[1]
	if seq1 == seq2:
		identity = 1
	else:
		bases = len(seq1)
		matches = 0
		for pos in range(bases):
			if seq1[pos] == seq2[pos]:
				matches += 1
		identity = matches / bases
	return identity

"""