#!/usr/bin/python3

# hw1.plot.1.py

# Shu-Ting Cho <shutingc@andrew.cmu.edu>

# plot for 5.2.5
# v1 2021/09/22

# import modules
import pandas as pd
import matplotlib.pyplot as plt


# Loading CSV Files (w/ imputation)
with_imputation = pd.read_csv("with_imputation.csv")
without_imputation = pd.read_csv("without_imputation.csv")

	
# Draw scatter plot
with_imputation.plot.scatter(x="# of training data", y="m_train", color='r', title='with imputation: train')
plt.savefig("with_imputation_train.png")

with_imputation.plot.scatter(x="# of training data", y="test", color='b', title='with imputation: test')
plt.savefig("with_imputation_test.png")


without_imputation.plot.scatter(x="# of training data", y="m_train", color='r', title='without imputation: train')
plt.savefig("without_imputation_train.png")

without_imputation.plot.scatter(x="# of training data", y="test", color='b', title='without imputation: test')
plt.savefig("without_imputation_test.png")