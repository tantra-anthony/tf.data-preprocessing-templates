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

# but we still need to split into training and test sets
# training set is where the model is built upon
# test set is where the model is tested for its accuracy
# import the train_test_split lib from sklearn.cross_validation
from sklearn.cross_validation import train_test_split

# create all training and test for X and y
# test_size is the percentage that goes to your test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================

# ML calculations are based on Euclidian distances (the y diff squared plus x diff squared)
# so every feature must be of the same scale, they must be of the same scale
# usually it's standardized using z standardization or normalization
# import StandardSclaer from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler

# apply it to the training set, need to both fit and transform
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

# since sc_X is already fitted to training, we don't fit it again to the test set
# this is to make sure that the scale for training set and test set is THE SAME
X_test = sc_X.transform(X_test)

# do we need to standardise dummy variables?
# either way is fine but it depends on the context
# feature scaling for dependent variable? no because this is only 2 values
# with many values then yes you need to do it