from IPython.display import display
from sklearn.metrics import confusion_matrix as _confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns



##### Plotting #####

def scatter_points(data, color=None, show_origin=False, equal_scale=False):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
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
    
    
    
def pair_plot(data):
    sns.pairplot(data)
    
    
    
#### Models ####

def linear_model(X, Y, add_constant=True, verbose=True):
    X_ext = sm.add_constant(X) if add_constant else X
    sm_mod = sm.OLS(Y, X_ext)
    sm_res = sm_mod.fit()
    if verbose:
        display(sm_res.summary())
    return sm_res


### Reports ###

def confusion_matrix(Y_true, Y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.append(Y_true, Y_pred))
    return pd.DataFrame(_confusion_matrix(Y_true, Y_pred, labels=labels),
                        index=[f'actual_{label}' for label in labels],
                        columns=[f'predicted_{label}' for label in labels])
    

