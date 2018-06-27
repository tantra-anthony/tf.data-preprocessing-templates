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

# to take care of missing data, we can average out all of the other values
# present in other data, this is also common practice
# import Imputer from sklearn.preprocessing to help

# =============================================================================

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

# for some of the data, we need mathematical representations of them
# therefore text need to be encoded into numbers
# here we use the sklearn library again to deal with this encoding
# import LabelEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# create labelencoder object for X
# then use fit_transform to 
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # returns an encoded array

# however there is still a problem, 0,1,2 should not have priority over another
# there is not supposed to be any order to the encoding algo
# to prevent this, we need a dummy variable
# thus we use a binary encoding table, use OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0]) # 0 is the index number of column

# fit onehotencoder to X
# don't have to specify index because we did it before
X = onehotencoder.fit_transform(X).toarray()

# now encode y
# note that if >2 forms apply you have to repeat the above using OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# data is now ready!

# =============================================================================
