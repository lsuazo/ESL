from IPython.display import display
from sklearn.metrics import confusion_matrix as _confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns



##### Plotting #####

def scatter_points(data, color=None):
    plt.scatter(data[:,0], data[:,1], c=color)
    
    
    
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
    

