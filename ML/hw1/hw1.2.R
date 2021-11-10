#!/usr/bin/python3

# hw1.2.R

# Shu-Ting Cho <shutingc@andrew.cmu.edu>
# In this problem, you will use census data that contains both categorical and continuous covariates to predict whether someone's income is >50K or <=50K. To do this you will be implementing KNN and Naive Bayes and comparing the difference between imputing and not imputing missing values.

# v1 2021/09/20

## First you will create new training and test data sets with imputed missing values. For this we will be using a variant of K-Nearest Neighbor (KNN) algorithm with k <- 10. 

k <- 10

# First, combine the training and test data, remove the label from each input sample and then divide the combined data into two sets. The first set contains all input samples (rows) with no missing values while the second contains all those with at least one missing value. 

setwd("C:/Users/vivia/OneDrive - University of Pittsburgh/Pitt/CPCB/ML/hw1/")

# Loading CSV Files
df_train <- read.delim("census.csv", header=TRUE, sep=",")
df_test <- read.delim("adult.test.csv", header=TRUE, sep=",")

# combine the training and test data
df_combine <- rbind(df_train, df_test)

# remove the label from each input sample
df_combine <- df_combine[,-ncol(df_combine)]

# divide the combined data into two sets. The first set contains all input samples (rows) with no missing values while the second contains all those with at least one missing value. 
# some cells without value were '' instead of NA, convert '' to NA
df_combine[df_combine==''] <- NA
df_combine_nona <- df_combine[rowSums(is.na(df_combine)) == 0,]
df_combine_na <- df_combine[rowSums(is.na(df_combine)) > 0,]

df_combine_na2 <- df_combine_na

## KNN
# Next, for each input sample in the second (missing values) set you would find its nearest neighbors in the first set. For this, we will use a distance metric to quantify how similar the rows are to each other.


# matrix to store distances with all rows in df_combine_nona
dist_df_h <- data.frame(matrix(nrow = nrow(df_combine_nona), ncol = 2))
rownames(dist_df_h) <- rownames(df_combine_nona)
colnames(dist_df_h) <- c("dist","rowname")

n <- 0
for (row_na in rownames(df_combine_na))
{
	n = n + 1
	print(n)
	# When calculating this metric you will use only the attributes that are not missing.
	# get cols for missing values, continuous and categorical attributes
	col_absent <- c()
	col_cont <- c()
	col_cate <- c()
	for (col_h in colnames(df_combine_na))
	{
		value_h <- df_combine_na[row_na, col_h]
		if (is.na(value_h)) {
			col_absent <- c(col_absent, col_h)
		} else if (is.numeric(value_h)) {
			col_cont <- c(col_cont, col_h)
		} else {
			col_cate <- c(col_cate, col_h)
		}
	}
	cont_q <- df_combine_na[row_na, col_cont]
	cate_q <- df_combine_na[row_na, col_cate]

	for (row_nona in rownames(df_combine_nona))
	{
		# For continuous attributes you will use Euclidean distance.
		cont_s <- c()
		for (col_h in col_cont)
		{
			cont_s <- c(cont_s, df_combine_nona[row_nona, col_h])
		}
		dist_cont <- stats::dist(rbind(cont_q, cont_s), method <- "euclidean")

		# For categorical attributes the distance will be 0 if the two categories are the same and 10 if they are different.
		dist_cate <- 0
		for (col_h in col_cate)
		{
			if (cate_q[col_h] != value_s) { 
				dist_cate <- dist_cate + 10
			}	
		}

		# To calculate the final distance just sum up the continuous and categorical distances for a pair of rows. 
		dist_df_h[row_nona, c("dist","rowname")] <- c(dist_cont + dist_cate, row_nona)
	}
	# Following this you can rank, for each row with missing values, all rows that do not have missing values based on their distance and chose the top K.
	sort_dist_df_h <- dist_df_h[order(dist_df_h$dist, decreasing=TRUE),]
	# rownames for top K
	K_rownames <- rownames(sort_dist_df_h)[c(1:k)]
	for (col_h in col_absent) {
		#print(table(df_combine_nona[K_rownames, col_h]), names(sort(-table(df_combine_nona[K_rownames, col_h])))[1])
		df_combine_na2[row_na, col_h] <- names(sort(-table(df_combine_nona[K_rownames, col_h])))[1]
	}
}





'''====================================================================================================='''



setwd("C:/Users/vivia/OneDrive - University of Pittsburgh/Pitt/CPCB/ML/hw1/")

# Loading CSV Files
df_train <- read.delim("census.csv", header=TRUE, sep=",")
df_test <- read.delim("adult.test.csv", header=TRUE, sep=",")

# combine the training and test data
df_combine <- rbind(df_train, df_test)

# remove the label from each input sample
df_combine <- df_combine[,-ncol(df_combine)]

# divide the combined data into two sets. The first set contains all input samples (rows) with no missing values while the second contains all those with at least one missing value. 
# some cells without value were '' instead of NA, convert '' to NA
df_combine[df_combine==''] <- NA
df_combine_nona <- df_combine[rowSums(is.na(df_combine)) == 0,]
df_combine_na <- df_combine[rowSums(is.na(df_combine)) > 0,]

df_combine_na2 <- df_combine_na

## KNN
# Next, for each input sample in the second (missing values) set you would find its nearest neighbors in the first set. For this, we will use a distance metric to quantify how similar the rows are to each other.

sub_df_combine_na <- df_combine_na[c(1:100),]
sub_df_combine_nona <- df_combine_nona[c(1:1000),]

# matrix to store distances with all rows in df_combine_nona
dist_df_h <- data.frame(matrix(nrow = nrow(sub_df_combine_nona), ncol = 2))
rownames(dist_df_h) <- rownames(sub_df_combine_nona)
colnames(dist_df_h) <- c("dist","rowname")

n <- 0
for (row_na in rownames(sub_df_combine_na))
{
	n = n + 1
	print(n)
	# When calculating this metric you will use only the attributes that are not missing.
	# get cols for missing values, continuous and categorical attributes
	col_absent <- c()
	col_cont <- c()
	col_cate <- c()
	for (col_h in colnames(sub_df_combine_na))
	{
		value_h <- sub_df_combine_na[row_na, col_h]
		if (is.na(value_h)) {
			col_absent <- c(col_absent, col_h)
		} else if (is.numeric(value_h)) {
			col_cont <- c(col_cont, col_h)
		} else {
			col_cate <- c(col_cate, col_h)
		}
	}
	cont_q <- sub_df_combine_na[row_na, col_cont]
	cate_q <- sub_df_combine_na[row_na, col_cate]

	for (row_nona in rownames(sub_df_combine_nona))
	{
		# For continuous attributes you will use Euclidean distance.
		cont_s <- c()
		for (col_h in col_cont)
		{
			cont_s <- c(cont_s, sub_df_combine_nona[row_nona, col_h])
		}
		dist_cont <- stats::dist(rbind(cont_q, cont_s), method <- "euclidean")

		# For categorical attributes the distance will be 0 if the two categories are the same and 10 if they are different.
		dist_cate <- 0
		for (col_h in col_cate)
		{
			if (cate_q[col_h] != value_s) { 
				dist_cate <- dist_cate + 10
			}	
		}

		# To calculate the final distance just sum up the continuous and categorical distances for a pair of rows. 
		dist_df_h[row_nona, c("dist","rowname")] <- c(dist_cont + dist_cate, row_nona)
	}
	# Following this you can rank, for each row with missing values, all rows that do not have missing values based on their distance and chose the top K.
	sort_dist_df_h <- dist_df_h[order(dist_df_h$dist, decreasing=TRUE),]
	# rownames for top K
	K_rownames <- rownames(sort_dist_df_h)[c(1:k)]
	for (col_h in col_absent) {
		#print(table(sub_df_combine_nona[K_rownames, col_h]), names(sort(-table(sub_df_combine_nona[K_rownames, col_h])))[1])
		df_combine_na2[row_na, col_h] <- names(sort(-table(sub_df_combine_nona[K_rownames, col_h])))[1]
	}
}










# Once you have the KNN for each missing value row you would impute the missing data as follows: For continuous values use the average of the values for this attribute in the KNN rows you identified. For categorical values use the majority value.

# In case of ties for categorical values (i.e. two or more categorical values have the same top number of appearances in the KNN), break them based on the total number of times the values appear in the entire dataset, choosing the category with the most number of occurrences. 