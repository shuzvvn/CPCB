#!/usr/bin/python3

# hw1.report.2.py

# Shu-Ting Cho <shutingc@andrew.cmu.edu>

# Instead of training the NB using all training data, train only with the first m data
# v1 2021/09/22

# import modules
import pandas as pd
from collections import Counter
import statistics
import math
import matplotlib.pyplot as plt


# Loading CSV Files (w/ imputation)
df_train = pd.read_csv("census.csv")
df_test = pd.read_csv("adult.test.csv")
df_train2 = pd.read_csv("df_train2.csv")
df_test2 = pd.read_csv("df_test2.csv")

# dict of df_data
dict_train = {"with_imputation": df_train2, "without_imputation": df_train}
dict_test = {"with_imputation": df_test2, "without_imputation": df_test}

dict_data_sets = {"with_imputation": {"train": df_train2, "test": df_test2}, "without_imputation": {"train": df_train, "test": df_test}}


############################################### 5.2.8 ###############################################
dict_df_report = {}
for run in dict_data_sets:
	dict_df_report[run] = pd.DataFrame()
	print("\n##### training data", run, "#####")
	train = dict_data_sets[run]["train"]
	test = dict_data_sets[run]["test"]
	
	# Select the first m data points including lines with missing values and call this your training data.
	for i in list(range(5,14)): # Repeat for m = {2 i for i = 5, 6, 7,..., 13} (i.e. m = 32,..., 8192)
		m = 2**i
		print("i=%i, m=%i" %(i, m))
		
		training_data = train.head(m)
		
		# Remove lines with missing values from your training data (so you have m − m' rows where m' rows contain missing values).
		training_data = training_data.dropna()
		
		# seperate data rich and poor 
		df_poor_h, df_rich_h = [x for _, x in training_data.groupby(training_data['income'] == " >50K")]
		dict_df_train = {">50K": df_rich_h, "<=50K": df_poor_h}
		
		# priors for y
		dict_prior = {}

		# likelihood
		dict_likelihood = {}
		list_cont = ["education-num", "capital-gain", "age", "capital-loss", "hours-per-week", "fnlwgt"]
		list_cate = ["marital-status", "race", "workclass", "occupation", "education", "relationship", "sex", "native-country"]

		nrow_train = training_data.shape[0]
		dict_df_report[run].at[m, "# of training data"] = nrow_train
		for y_h in dict_df_train:
			# priors
			nrow_y = dict_df_train[y_h].shape[0]
			dict_prior[y_h] = nrow_y / nrow_train

			# likelihood
			dict_likelihood[y_h] = {} # parameters for >50 or <=50
			
			# continuous: mean and variance
			for feature_h in list_cont:
				mean_h = statistics.mean(dict_df_train[y_h][feature_h])
				var_h = statistics.pvariance(dict_df_train[y_h][feature_h])
				dict_likelihood[y_h][feature_h] = [mean_h, var_h]

			# discrete, P(xi=j|Y=c)
			for feature_h in list_cate:
				dict_count = dict(Counter(dict_df_train[y_h][feature_h]))
				dict_likelihood[y_h][feature_h] = {}
				for j in dict_count:
					dict_likelihood[y_h][feature_h][j] = dict_count[j]/nrow_y

		# accuracy
		dict_test_h = {"m_train": training_data, "test": test}
		for test_h in dict_test_h:
			n = 0
			n_correct = 0
			for row_h in dict_test_h[test_h].index.tolist():
				n += 1
				posteriors = []
				for y_h in dict_df_train: # >50K, <=50K
					posterior_log_pyX = 0 # sum of log(P(xi|y)) + log(P(y))
					for col_h in dict_df_train[y_h].columns[:14]: # features
						# calculate P(xi|y)
						value_h = dict_test_h[test_h].at[row_h, col_h]
						if not pd.isna(value_h):
							# cont or cate
							try: # cont
								value_h = float(value_h)
								# mean and var from dict_likelihood[y_h][col_h]
								mean_h = dict_likelihood[y_h][col_h][0]
								var_h = dict_likelihood[y_h][col_h][1]+10**-9 # var + epsilon
								p_value_h = (1 / math.sqrt(2*math.pi*var_h)) * math.exp(-(((value_h-mean_h)**2)/(2*var_h)))
							except: # cate
								value_h = str(value_h)
								try:
									p_value_h = dict_likelihood[y_h][col_h][value_h]
								except: # value_h was not found in Y=y data set
									p_value_h = 0						
							# add log value to the sum
							if p_value_h != 0:
								p_value_h = math.log(p_value_h)
							else:
								p_value_h = -math.inf
							posterior_log_pyX = posterior_log_pyX + p_value_h
					posterior_log_pyX = math.log(dict_prior[y_h]) + posterior_log_pyX
					dict_test_h[test_h].at[row_h,y_h] = posterior_log_pyX
					posteriors.append(posterior_log_pyX)
				if posteriors[0] < posteriors[1]: # posteriors(">50K") < posteriors("<=50K")
					prediction = "<=50K"
				else:
					prediction = ">50K"
				dict_test_h[test_h].at[row_h,"prediction"] = prediction
				if prediction == dict_test_h[test_h].at[row_h, "income"].strip():
					n_correct += 1
			accuracy = n_correct/n
			#print("%s: %.4f" % (test_h, accuracy))
			dict_df_report[run].at[m, test_h] = round(accuracy, 4)
	print(dict_df_report[run])

dict_df_report["with_imputation"].to_csv(r'with_imputation.csv', index = True)
dict_df_report["without_imputation"].to_csv(r'without_imputation.csv', index = True)