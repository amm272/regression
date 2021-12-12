#%%
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge
import seaborn as sns

# Generate test datasets
def fit(xtrain, ytrain, xtest, ytest):
    regr = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
    init = [0.02,0.1] #Those parameters influence the behaviour of the model
    regr.set_params(alpha_init=init[0], lambda_init=init[1])

    # Fit model
    regr.fit(xtrain.reshape(-1,1),ytrain)
    print(xtrain.reshape(-1,1))
    print(ytrain)

    #%%
    predictions, std = regr.predict(xtest.reshape(-1,1), return_std=True)

    #%%
    # Display plots with predictions
    plt.scatter(xtrain, ytrain, s=50, alpha=0.5, label="observation")
    plt.plot(xtest, predictions, color="red", label="predict mean")
    plt.fill_between(xtest, predictions-std, predictions+std, color="pink", alpha=0.5, label="predict std")
    plt.show()


    #%%
    # Evaluate the performance of the model
    print('Coefficients: \n', regr.coef_)
    print('Coefficient of determination: \.2f', r2_score(ytest,predictions))

    # Validate the model using residuals plot
    sns.residplot(x=ytest,y=predictions,lowess=True,color="g")
    plt.show()
