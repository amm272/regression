
#%%
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge
import seaborn as sns
from utils.regression_utils import *

# Generate test datasets
def fit(xtrain, ytrain, xtest, ytest):
    # Initial parameters - figure out how to estimate these
    alpha = 0.01
    beta = 0.5
    # Training observations data
    X = xtrain.reshape(-1, 1)
    length = len(xtrain)
    print(length)

    # Training target values
    t = ytrain

    # Test observations
    X_test = xtest.reshape(-1, 1)

    # Ground truth function values
    y_true = ytest

    # Design matrix of test observations
    Phi_test = expand(X_test, polynomial_basis_function)

    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4)

    # Design matrix of training observations
    Phi_N = expand(X, polynomial_basis_function)

    # Mean and covariance matrix of posterior
    m_N, S_N = posterior(Phi_N, t, alpha, beta)

    # Mean and variances of posterior predictive 
    y, y_var = posterior_predictive(Phi_test, m_N, S_N, beta)
  
    # Draw 5 random weight samples from posterior and compute y values
    w_samples = np.random.multivariate_normal(m_N.ravel(), S_N, 5).T
    y_samples = Phi_test.dot(w_samples)

    # Initial points
    f_w0 = 0
    f_w1 = 0

    plt.subplot(1, 3, 1)
    plot_posterior(m_N, S_N, f_w0, f_w1)
    plt.title(f'Posterior density')
    plt.legend()

    plt.subplot(1, 3, 2)
    plot_data(X, t)
    plot_truth(X_test, y_true)
    plot_posterior_samples(X_test, y_samples)
    plt.ylim(-1.5, 1.0)
    plt.legend()

    plt.subplot(1, 3, 3)
    plot_data(X, t)
    plot_truth(X_test, y_true, label=None)
    plot_predictive(X_test, y, np.sqrt(y_var))
    plt.ylim(-1.5, 1.0)
    plt.legend()
# %%
