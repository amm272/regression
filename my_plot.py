
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import regression_utils
from models import bayesian_linear_regression, bayesian_poly_regression

# Import Datasets
df = regression_utils.import_data('Data_FCR.csv', 'plot.py', 15)
X = df[['X1']]

# Split my data into training and testing
# Training data
x = df.X1.values
y = df.Y.values
l = len(x)
itrain, itest = train_test_split(np.arange(l), train_size=0.4)
xtrain = x[itrain]
ytrain = y[itrain]

plt.plot(xtrain,ytrain, 'o')
plt.show()

# Test data
xtest= x[itest]
ytest = y[itest]

plt.plot(xtest,ytest, 'o')
plt.show()

# Compare different models
# Bayesian Ridge Regression model
# bayesian_linear_regression.fit(xtrain, ytrain, xtest, ytest)

# Bayesian Ploynomial Model
bayesian_poly_regression.fit(xtrain, ytrain, xtest, ytest)
# %%
