from IPython.display import display
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.model_selection import train_test_split as ttsplit


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns



##### Plotting #####

def scatter_points(ax, data, color=None, show_origin=False, equal_scale=False):
    x = data[:,0]
    y = data[:,1]
    
    if equal_scale:
        ax.set_aspect('equal', 'box')
    
    if show_origin:
        padx = (x.max() - x.min())/10
        pady = (y.max() - y.min())/10

        xmin = np.min([x.min()-padx, -1])
        xmax = np.max([x.max()+padx, 1])
        ymin = np.min([y.min()-pady, -1])
        ymax = np.max([y.max()+pady, 1])

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        ax.axhline(0, linestyle='--', alpha=0.25, color='k')
        ax.axvline(0, linestyle='--', alpha=0.25, color='k')
    
    ax.scatter(x, y, c=color)
    
def plot_infinite_line(ax, p0, p1):
    slope = (p1[1] - p0[1])/(p1[0] - p0[1])
    xmin, xmax = ax.get_xlim()
    ax.plot([xmin, xmax], [xmin*slope, xmax*slope], color='r', alpha=0.25)
    
def pair_plot(data):
    sns.pairplot(data)
    
    

#### Data Processing####
def demean(X):
    return X - X.mean(axis=0)

def train_test_split(X,Y, test_fraction, random_seed=None):
    return ttsplit(X,Y,test_size=test_fraction, random_state = random_seed)
    
#### Models ####

def linear_model(X, Y, add_constant=True, verbose=True):
    X_ext = sm.add_constant(X) if add_constant else X
    sm_mod = sm.OLS(Y, X_ext)
    sm_res = sm_mod.fit()
    if verbose:
        display(sm_res.summary())
    return sm_res

def regularized_linear_model(X, Y, method, alpha, l1_ratio=None, add_constant=True, verbose=True):
    available_methods = ['ridge', 'lasso', 'elastic_net']
    method_ = method.lower()
    assert method_ in available_methods, f"Must be in {available_methods}"
    if method_ == 'ridge':
        model = Ridge(alpha=alpha)
    elif method == 'lasso':
        model = Lasso(alpha=alpha)
    elif method == 'elastic_net':
        l1 is l1_ratio if l1_ratio is not None else 0.5
        model = ElasticNet(alpha = alpha, l1_ratio = l1)
    else:
        raise NotImplementedError
    model.fit(X,Y)
    return model


### Reports ###

def confusion_matrix(Y_true, Y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.append(Y_true, Y_pred))
    return pd.DataFrame(_confusion_matrix(Y_true, Y_pred, labels=labels),
                        index=[f'actual_{label}' for label in labels],
                        columns=[f'predicted_{label}' for label in labels])
    

