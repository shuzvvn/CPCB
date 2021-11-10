#!/usr/bin/python3

# ML_hw2.1.py

# Shu-Ting Cho <shutingc@andrew.cmu.edu>
# Data preprocessing

# v1 2021/09/29


# start time
import time
print(time.strftime("%H:%M:%S", time.localtime()), "start")


# import modules
import pandas as pd


# Loading CSV Files
d_train_0 = pd.read_csv("carseats_train.csv")
d_test_0 = pd.read_csv("carseats_test.csv")



########## 1. Binary variable encoding ##########
# In this dataset, Urban, US are both binary variables which take values No and Yes. You must convert them to 0/1 binary variables so that these numerical values can be used for linear regression.

datasets_1 = {"train": d_train_0, "test": d_test_0}

for data in datasets_1:
	datasets_1[data] = datasets_1[data].replace(['Yes'], 1)
	datasets_1[data] = datasets_1[data].replace(['No'], 0)


########## 2. Categorical variable encoding ##########
# ShelveLoc is a categorical variable that takes three values, namely Bad, Good, Medium. You must do one-hot encoding for this particular variable. This means that you must create three dummy variables: ShelveLocBad, ShelveLocGood, ShelveLocMedium. ShelveLocBad should be 1 when the value of ShelveLoc is Bad and 0 otherwise. ShelveLocGood, ShelveLocMedium should be encoded in a similar way. This means that for any datapoint, exactly one of ShelveLocBad, ShelveLocGood, ShelveLocMedium will take the value 1 and the other two will be 0.

datasets_2 = {"train": datasets_1["train"], "test": datasets_1["test"]}

for data in datasets_2:
	datasets_2[data] = datasets_2[data].assign(ShelveLocBad=0, ShelveLocGood=0, ShelveLocMedium=0)
	for row in datasets_2[data].index:
		if datasets_2[data].at[row, "ShelveLoc"] == "Bad":
			datasets_2[data].at[row, "ShelveLocBad"] = 1
		elif datasets_2[data].at[row, "ShelveLoc"] == "Good":
			datasets_2[data].at[row, "ShelveLocGood"] = 1
		else:
			datasets_2[data].at[row, "ShelveLocMedium"] = 1
	datasets_2[data] = datasets_2[data].drop('ShelveLoc', 1)


########## 3. Feature standardization ##########
# Feature standardization makes the data such that it has zero mean and unit variance. For every continuous covariate, you must subtract the mean and divide it by the standard deviation.
# NOTE: You must do the feature standardization for the test set using the mean and standard deviation calculated from the train set.

datasets_3 = {"train": datasets_2["train"], "test": datasets_2["test"]}
cont_cols = ["CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"]

for col in cont_cols:
	col_mean = datasets_3["train"][col].mean()
	col_stdev = datasets_3["train"][col].std(ddof=0)
	for data in datasets_3: # use the mean and std from train to standardize train and test data
		datasets_3[data][col] = (datasets_3[data][col] - col_mean) / col_stdev


datasets_3["train"].to_csv(r'datasets_3_train.csv', index = False)
datasets_3["test"].to_csv(r'datasets_3_test.csv', index = False)


print(time.strftime("%H:%M:%S", time.localtime()), "end")