#from optparse import Values
import numpy as np
import math
import pandas as pd
def entropy(Y):

   # Function to calculate the entropy #
    N_labels=len(Y)
    if(N_labels<1):
        return 0
    value,counts = np.unique(Y, return_counts=True)

    probs = counts / N_labels
    

    
    if len(value) <= 1:
        return 0

    entopy = 0

  # Compute entropy

    for i in probs:
        entopy += (-i)*math.log2(i)

    return entopy
    
    pass

def gini_index(Y):
    """
    Function to calculate the gini index
    
    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    N_labels=len(Y)
    if(N_labels<1):
        return 0
    value,counts = np.unique(Y, return_counts=True)
    gini_indx=1
    for i in range(len(value)):
        p=counts[i]/float(len(Y))
        gini_indx=gini_indx-p**2
    
    return gini_indx
    pass

def information_gain(Y, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    Y.name="Target"
    attr.name="attribute"
    total_entropy = entropy(Y)
    weigh_entr=0
    
    ##Calculate the entropy of the dataset
    
    #Calculate the values and the corresponding counts for the split attribute 
    value,counts= np.unique(attr,return_counts=True)
    df=pd.concat([attr,Y],axis=1)
    total_size=len(Y)
    for i in value:
        df1=df[df["attribute"]==i]
        cur_size=df1.shape[0]#number of rows
        weigh_entr+=(cur_size/float(total_size))*entropy(df1["Target"])
    #Calculate the weighted entropy
    return(total_entropy-weigh_entr)

            
            

        #if()
    #Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(y)])
    
    #Calculate the information gain
    
    pass
def gini_gain(Y, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    Y.name="Target"
    attr.name="attribute"
    total_gini = gini_index(Y)
    weigh_gini=0
    
    ##Calculate the entropy of the dataset
    
    #Calculate the values and the corresponding counts for the split attribute 
    value,counts= np.unique(attr,return_counts=True)
    df=pd.concat([attr,Y],axis=1)
    total_size=len(Y)
    for i in value:
        df1=df[df["attribute"]==i]
        cur_size=df1.shape[0]#number of rows
        weigh_gini+=(cur_size/float(total_size))*gini_index(df1["Target"])
    #Calculate the weighted entropy
    return(total_gini-weigh_gini)

            
            
