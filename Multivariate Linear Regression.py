#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


#define the number of layers
#one = 2:13
#five = 14:35
#ten = 37:63
#fifteen = 66:92

df = pandas.read_csv("Data_FCR.csv")
df = df[66:92]

print(df.head())

X = df[['X1']]
x = df.X1.values
y = df.Y.values
l = len(x)

#%%
#Split my data into training and testing
itrain, itest = train_test_split(np.arange(l), train_size=0.8)
xtrain = x[itrain]
ytrain = y[itrain]

plt.plot(xtrain,ytrain, 'o')

#%%
xtest= x[itest]
ytest = y[itest]

plt.plot(xtest,ytest, 'o')

#%%
# Implement bootstrapping algorithm
N = 1000
bootstrap_beta1s = np.zeros(N)
for cur_bootstrap_rep in range(N):
    # select indices that are in the resample (easiest way to be sure we grab y values that match the x values)
    inds_to_sample = np.random.choice(xtrain.shape[0], size=xtrain.shape[0], replace=True)
    
    # take the sample
    x_train_resample = xtrain[inds_to_sample]
    y_train_resample = ytrain[inds_to_sample]
    
    # fit the model
    bootstrap_model = LinearRegression().fit(x_train_resample.reshape(-1,1), y_train_resample)
    
    # extract the beta1 and append
    bootstrap_beta1s[cur_bootstrap_rep] = bootstrap_model.coef_[0]

## display the results

# calculate 5th and 95th percentiles
lower_limit, upper_limit = np.percentile(bootstrap_beta1s,[5,95])

# plot histogram and bounds
fig, ax = plt.subplots(1,1, figsize=(20,10))
ax.hist(bootstrap_beta1s, 20, alpha=0.6, label=r"Bootstrapped $\beta_{1}$ values")
ax.axvline(lower_limit, color='red', label=r"$5$th Percentile ({:.2f})".format(lower_limit))
ax.axvline(upper_limit, color='black', label=r"$95$th Percentile ({:.2f})".format(upper_limit))

# good plots have labels
ax.set_xlabel(r"$\beta_{1}$ Values")
ax.set_ylabel("Count (out of 1000 Bootstrap Replications)")
plt.title(r"Bootstrapped Values of $\beta_{1}$")
plt.legend();

#%%
# Run ridge regression for simple linear regression
regression_coeffs = dict() # Store regression coefficients from each model in a dictionary

regression_coeffs['OLS'] = [np.nan]*2 # Initialize to NaN
regression_coeffs[r'Ridge $\lambda = 0$'] = [np.nan]*2

dfResults = pd.DataFrame(regression_coeffs) # Create dataframe

dfResults.rename({0: r'$\beta_{0}$', 1: r'$\beta_{1}$'}, inplace=True) # Rename rows
dfResults

#%%
# Implement simple linear regression
simp_reg = LinearRegression() # build the the ordinary least squares model

simp_reg.fit(xtrain.reshape(-1,1), ytrain) # fit the model to training data

# save the beta coefficients
beta0_sreg = simp_reg.intercept_
beta1_sreg = simp_reg.coef_[0]

dfResults['OLS'][:] = [beta0_sreg, beta1_sreg]
dfResults

# y_predict = lambda x : beta0_sreg + beta1_sreg*x # make predictions
ypredict_ols = simp_reg.predict(x.reshape(-1,1))
ypredict_ols.shape

#%%
ridge_reg = Ridge(alpha = 0) # build the ridge regression model with specified lambda, i.e. alpha

ridge_reg.fit(xtrain.reshape(-1,1), ytrain) # fit the model to training data

# save the beta coefficients
beta0_ridge = ridge_reg.intercept_
beta1_ridge = ridge_reg.coef_[0]

ypredict_ridge = ridge_reg.predict(x.reshape(-1,1)) # make predictions everywhere

dfResults[r'Ridge $\lambda = 0$'][:] = [beta0_ridge, beta1_ridge]
dfResults

#%%
fig, ax = plt.subplots(1,1, figsize=(10,6))

ax.plot(xtrain, ytrain, 's', alpha=0.3, ms=10, label="in-sample y (observed)") # plot in-sample training data
ax.plot(x, y, '.', alpha=0.4, label="population y") # plot population data
ax.plot(x, ypredict_ols, ls='--', lw=4, label="OLS") # plot simple linear regression fit
ax.plot(x, ypredict_ridge, ls='-.', lw = 4, label="Ridge") # plot ridge regression fit

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.legend(loc=4);

fig.tight_layout()

#%%
# Your code here
fig, ax = plt.subplots(1,1, figsize=(20,10))

pen_params = [0, 0.0000005, 0.000001, 0.000002, 0.000003, 0.000004, 0.00001]
ax.plot(xtrain, ytrain, 's', alpha=0.5, ms=10, label="in-sample y (observed)") # plot in-sample training data

for alpha in pen_params:
    ridge_reg = Ridge(alpha = alpha) # build the ridge regression model with specified lambda, i.e. alpha
    ridge_reg.fit(xtrain.reshape(-1,1), ytrain) # fit the model to training data
    ypredict_ridge = ridge_reg.predict(x.reshape(-1,1))
    ax.plot(x, ypredict_ridge, ls='-.', lw = 4, label=r"$\lambda = {}$".format(alpha)) # plot ridge regression fit

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.legend(loc=4, fontsize=24);

fig.tight_layout()

fig.savefig('ridge_lambda.png')

#%%
d = 7 # Maximum polynomial degree
# You will create a grid of plots of this size (3 x 2)
rows = 4
cols = 2

#%%
lambdas = [0, 1e-7, 5*1e-7,1e-6] # Various penalization parameters to try
grid_to_predict = np.arange(0.0002, 0.00141, .0001) # Predictions will be made on this grid

# Create training set and test set
Xtrain = PolynomialFeatures(d).fit_transform(xtrain.reshape(-1,1))
test_set = PolynomialFeatures(d).fit_transform(grid_to_predict.reshape(-1,1))

fig, axs = plt.subplots(rows, cols, sharex='col', figsize=(12, 24)) # Set up plotting objects

for i, lam in enumerate(lambdas):
    # your code here
    ridge_reg = Ridge(alpha = lam) # Create regression object
    ridge_reg.fit(Xtrain, ytrain) # Fit on regression object
    ypredict_ridge = ridge_reg.predict(test_set) # Do a prediction on the test set
    
    ### Provided code
    axs[i,0].plot(xtrain, ytrain, 's', alpha=0.4, ms=10, label="in-sample y") # Plot sample observations
    axs[i,0].plot(grid_to_predict, ypredict_ridge, 'k-', label=r"$\lambda =  {0}$".format(lam)) # Ridge regression prediction
    axs[i,0].set_ylabel('$y$') # y axis label
    #axs[i,0].set_ylim((0, 1)) # y axis limits
    axs[i,0].set_xlim((0, 0.0014)) # x axis limits
    axs[i,0].legend(loc='best') # legend
    
    coef = ridge_reg.coef_.ravel() # Unpack the coefficients from the regression
    
    axs[i,1].semilogy(np.abs(coef), ls=' ', marker='o', label=r"$\lambda =  {0}$".format(lam)) # plot coefficients
    #axs[i,1].set_ylim((1e-04, 1e+15)) # Set y axis limits
    #axs[i,1].set_xlim(1, 20) # Set y axis limits
    #axs[i,1].yaxis.set_label_position("right") # Move y-axis label to right
    #axs[i,1].set_ylabel(r'$\left|\beta_{j}\right|$') # Label y-axis
    #axs[i,1].legend(loc='best') # Legend

# Label x axes
#axs[-1, 0].set_xlabel("x")
#axs[-1, 1].set_xlabel(r"$j$");

#%%
#Plot a scatter plot between the dependent and the independent variables
df.plot('X1', 'Y', kind = 'scatter') #R versus strain
df.plot('X2', 'Y', kind = 'scatter') #R versus number of layers
df.plot('X1', 'X2', kind = 'scatter') #Strain versus number of layers
plt.show()

#Choose the most appropriate model
#Linear model
#regr = linear_model.LinearRegression()

#Bayesian Ridge Regression model
regr = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
init = [0.02,0.1]
regr.set_params(alpha_init=init[0], lambda_init=init[1])

#Ridge Regression model
#regr = Ridge(alpha=0)

#Fit the most suitable model
regr.fit(X,y)
predictions, std = regr.predict(X, return_std=True)

#Evaluate the performance of the model
print('Coefficients: \n', regr.coef_)
print('Coefficient of determination: \.2f', r2_score(y,predictions))

#Validate the model using residuals plot
sns.residplot(x=y,y=predictions,lowess=True,color="g")
plt.show()

#Make model plots for each of the samples with 1, 5, 10 and 15 layers
#that indicate bounding lines for area which includes 95% of the variance
plt.scatter(X, y, s=50, alpha=0.5, label="observation")
plt.plot(X, predictions, color="red", label="predict mean")
plt.fill_between(x, predictions-std, predictions+std, color="pink", alpha=0.5, label="predict std")
# %%
