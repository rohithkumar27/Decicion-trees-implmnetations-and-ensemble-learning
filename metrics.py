import numpy as np
import pandas as pd
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """

    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """

    #assert(len(y_hat) == len(y))
    correct=0
    y_np=y.to_numpy()
    y_hat_np=y_hat.to_numpy()
    for i in range(len(y)):
        if(y_np[i]==y_hat_np[i]):
            correct=correct+1
    return correct/float(len(y)) *100       
    # TODO: Write here
    pass

def precision(y_hat, y, cls):
    """
    Function to calculate the precision
    
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    y_np=y.to_numpy()
    y_hat_np=y_hat.to_numpy()
    numer=0
    denom=0
    for i in range(len(y_np)):
        if(y_np[i]==y_hat_np[i] and y_np[i]==cls):
           numer=numer+1
        if(y_np[i]==cls) :
            denom=denom+1         
    
    return numer/float(denom)
    pass

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    y_np=y.to_numpy()
    y_hat_np=y_hat.to_numpy()
    numer=0
    denom=0
    for i in range(len(y_np)):
        if(y_np[i]==y_hat_np[i] ):
            if(y_np[i]==cls):
                numer=numer+1
        if(y_hat_np[i]==cls) :
            denom=denom+1         
    
    return numer/float(denom)
    pass


def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    y_np=y.to_numpy()
    y_hat_np=y_hat.to_numpy()
    return np.sqrt(((y_hat_np - y_np) ** 2).mean())
    
    pass

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    y_np=y.to_numpy()
    y_hat_np=y_hat.to_numpy()
    return np.absolute(y_hat_np - y_np).mean()
    pass
