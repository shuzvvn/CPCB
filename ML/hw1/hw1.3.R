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
colnames(dist_df_h) <- c("dist")
dist_df_h$X <- ""

n <- 0
for (row_na in rownames(df_combine_na))
{
	n = n + 1
	print(n)
	for (row_nona in rownames(df_combine_nona))
	{
		cont_q <- c()
		cont_s <- c()
		dist_cate <- 0
		col_miss <- c()
		for (col_h in colnames(df_combine_na))
		{
			value_q <- df_combine_na[row_na, col_h]
			value_s <- df_combine_nona[row_nona, col_h]
			if (!is.na(value_q)) {
				if (is.numeric(value_q)) {
					cont_q <- c(cont_q, value_q)
					cont_s <- c(cont_s, value_s)
				} else if (value_q != value_s) {
					dist_cate <- dist_cate + 10
				}
			} else {
				col_miss <- c(col_miss, col_h)
			}
		}

		dist_cont <- stats::dist(rbind(cont_q, cont_s), method <- "euclidean")

		# To calculate the final distance just sum up the continuous and categorical distances for a pair of rows. 
		dist_df_h[row_nona, "dist"] <- dist_cont + dist_cate
	}
	# Following this you can rank, for each row with missing values, all rows that do not have missing values based on their distance and chose the top K.
	sort_dist_df_h <- dist_df_h[order(dist_df_h$dist, decreasing=TRUE),]
	# rownames for top K
	K_rownames <- rownames(sort_dist_df_h)[c(1:k)]
	col_miss <- unique(col_miss)
	for (col_h in col_miss) {
		df_combine_na2[row_na, col_h] <- names(sort(-table(df_combine_nona[K_rownames, col_h])))[1]
	}
}



'''============================================================================================================='''
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

# matrix to store distances with all rows in sub_df_combine_nona
dist_df_h <- data.frame(matrix(nrow = nrow(sub_df_combine_nona), ncol = 2))
rownames(dist_df_h) <- rownames(sub_df_combine_nona)
colnames(dist_df_h) <- c("dist")
dist_df_h$X <- ""

n <- 0
for (row_na in rownames(sub_df_combine_na))
{
	n = n + 1
	print(n)
	for (row_nona in rownames(sub_df_combine_nona))
	{
		cont_q <- c()
		cont_s <- c()
		dist_cate <- 0
		col_miss <- c()
		for (col_h in colnames(sub_df_combine_na))
		{
			value_q <- sub_df_combine_na[row_na, col_h]
			value_s <- sub_df_combine_nona[row_nona, col_h]
			if (!is.na(value_q)) {
				if (is.numeric(value_q)) {
					cont_q <- c(cont_q, value_q)
					cont_s <- c(cont_s, value_s)
				} else if (value_q != value_s) {
					dist_cate <- dist_cate + 10
				}
			} else {
				col_miss <- c(col_miss, col_h)
			}
		}

		dist_cont <- stats::dist(rbind(cont_q, cont_s), method <- "euclidean")

		# To calculate the final distance just sum up the continuous and categorical distances for a pair of rows. 
		dist_df_h[row_nona, "dist"] <- dist_cont + dist_cate
	}
	# Following this you can rank, for each row with missing values, all rows that do not have missing values based on their distance and chose the top K.
	sort_dist_df_h <- dist_df_h[order(dist_df_h$dist, decreasing=TRUE),]
	# rownames for top K
	K_rownames <- rownames(sort_dist_df_h)[c(1:k)]
	col_miss <- unique(col_miss)
	for (col_h in col_miss) {
		df_combine_na2[row_na, col_h] <- names(sort(-table(sub_df_combine_nona[K_rownames, col_h])))[1]
	}
}
