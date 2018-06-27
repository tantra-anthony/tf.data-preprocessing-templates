# This is a template for data preprocessing

# import numpy for mathematical operations and working with arrays
# import matplotlib to plot graphs which helps us illustrate the result
# import pandas to manage datasets (usually from csv)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the datasets using pandas, use read_csv for csv or any related format
dataset = pd.read_csv('Your File Here.csv')

# put dependent variable in variable "y" it's common practice
# put independent variable in variable "X" (capital)

# use pandas to extract the indexes column/row index start from 0
# first : is to get every column, -1 is index of last column
# this is the independent variable
X = dataset.iloc[:, :-1].values

# this is the dependent variable
# only take the last one
y = dataset.iloc[:, -1].values

# =============================================================================

# to take care of missing data, we can average out all of the other values
# present in other data, this is also common practice
# import Imputer from sklearn.preprocessing to help

from sklearn.preprocessing import Imputer

# create variable for Imputer, use Ctrl + I to inspect parameters
# first parameter is the values of the missing fields, typically NaN
# second parameter is the method we're going to use to replace missing val
# third parameter is axis, where we either take from column or row
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

# fit imputer to matrix object X
# however, only fit imputer to columns with missing data ONLY
imputer.fit(X[:, 1:3])

# replace missing data with mean, transform replaces the data
X[:, 1:3] = imputer.transform(X[:, 1:3])

# =============================================================================