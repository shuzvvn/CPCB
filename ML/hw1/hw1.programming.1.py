#!/usr/bin/python3

# hw1.program.1.py

# Shu-Ting Cho <shutingc@andrew.cmu.edu>

#### In this problem, you will use census data that contains both categorical and continuous covariates to predict whether someone's income is >50K or <=50K. To do this you will be implementing KNN and Naive Bayes and comparing the difference between imputing and not imputing missing values.

# imputation
# v1 2021/09/20

# import modules
import pandas as pd
from collections import Counter
import statistics
import math


### First you will create new training and test data sets with imputed missing values. For this we will be using a variant of K-Nearest Neighbor (KNN) algorithm with k = 10. 
k = 10

## First, combine the training and test data, remove the label from each input sample and then divide the combined data into two sets. The first set contains all input samples (rows) with no missing values while the second contains all those with at least one missing value. 


# Loading CSV Files
df_train = pd.read_csv("census.csv")
df_test = pd.read_csv("adult.test.csv")

# add data set name
df_train = df_train.assign(dataset="train")
df_test = df_test.assign(dataset="test")


# combine the training and test data
data_combine = df_train.append(df_test)
data_combine = data_combine.set_axis(list(range(1,len(data_combine)+1)))


# clone data_combine for imputation
data_combine2 = data_combine

# remove the label and data set name cols
data_combine = data_combine.iloc[:, :-2]


## divide the combined data into two sets. The first set contains all input samples (rows) with no missing values while the second contains all those with at least one missing value. 

# w/ NaN
data_combine_na = data_combine[data_combine.isnull().any(axis=1)]

# w/o NaN, drop rows which contain any NaN values
data_combine_nona = data_combine.dropna()


################## imputation KNN ##################
# Next, for each input sample in the second (missing values) set you would find its nearest neighbors in the first set. For this, we will use a distance metric to quantify how similar the rows are to each other.

for row_na in data_combine_na.index.tolist():
	# When calculating this metric you will use only the attributes that are not missing.
	# get cols for missing values
	col_missing = []

	# create an Empty DataFrame to store distances
	df_dist = pd.DataFrame()
	
	# get distance for each col, store in df
	for col_h in data_combine_na.columns:
		value_h = data_combine_na.at[row_na, col_h]
		if pd.isna(value_h): # store colname of missing value
			col_missing.append(col_h)
		else:
			try:
				# For continuous attributes you will use Euclidean distance.
				value_h = float(value_h)
				dist = abs(data_combine_nona[col_h] - value_h)
				df_dist[col_h] = dist
			except:
				# For categorical attributes the distance will be 0 if the two categories are the same and 10 if they are different.
				value_h = str(value_h)
				df_dist[col_h] = (data_combine_nona[col_h] != value_h) * 10
	
	# To calculate the final distance just sum up the continuous and categorical distances for a pair of rows. 
	df_dist['sum'] = df_dist.sum(axis=1)
	
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

## After imputation, you will implement the Naive Bayes (NB) algorithm on both the original training data set and the new training data set with imputed values.
# seperate train and test
df_train2, df_test2 = [x for _, x in data_combine2.groupby(data_combine2['dataset'] == "test")]
# df_train2.to_csv(r'df_train2.csv', index = False)
# df_test2.to_csv(r'df_test2.csv', index = False)

############################# report ############################
# post-imputation
# For questions below, report only up to 4 significant digits after the decimal points. In addition, for the questions below in this section use the data set with imputation of missing values.
# v1 2021/09/22

# Loading CSV Files (w/ imputation)
# df_train2 = pd.read_csv("df_train2.csv")
# df_test2 = pd.read_csv("df_test2.csv")

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
# df_train = pd.read_csv("census.csv")
# df_test = pd.read_csv("adult.test.csv")
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

# dict_df_report["with_imputation"].to_csv(r'with_imputation.csv', index = True)
# dict_df_report["without_imputation"].to_csv(r'without_imputation.csv', index = True)



############################# plot ##############################

# plot for 5.2.5
# v1 2021/09/22

# import modules
import pandas as pd
import matplotlib.pyplot as plt


# # Loading CSV Files (w/ imputation)
# with_imputation = pd.read_csv("with_imputation.csv")
# without_imputation = pd.read_csv("without_imputation.csv")

	
# Draw scatter plot
dict_df_report["with_imputation"].plot.scatter(x="# of training data", y="m_train", color='r', title='with imputation: train')
plt.savefig("with_imputation_train.png")

dict_df_report["with_imputation"].plot.scatter(x="# of training data", y="test", color='b', title='with imputation: test')
plt.savefig("with_imputation_test.png")


dict_df_report["without_imputation"].plot.scatter(x="# of training data", y="m_train", color='r', title='without imputation: train')
plt.savefig("without_imputation_train.png")

dict_df_report["without_imputation"].plot.scatter(x="# of training data", y="test", color='b', title='without imputation: test')
plt.savefig("without_imputation_test.png")