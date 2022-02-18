"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import column
from .utils import entropy,information_gain,gini_index,gini_gain

#import treenode as t
import math
#np.random.seed(42)      
class TreeNode:
    def __init__(self, data,output):
        # data represents the feature upon which the node was split when fitting the training data
        # data = None for leaf node
        self.data = data
        # children of a node are stored as a dicticionary with key being the value of feature upon which the node was split
        # and the corresponding value stores the child TreeNode
        self.children = {}
        # output represents the class with current majority at this instance of the decision tree
        self.output = output
        # index will be used to assign a unique index to each node
        self.index = -1
        
    def add_child(self,feature_value,obj):
        self.children[feature_value] = obj
class DecisionTree3():
    def __init__(self, criterion, max_depth):

        self.criterion=criterion
        self.max_depth=max_depth
        self.root=None

            
        
        
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        
        pass


    def grow_tree(self,X,y,features,level):
        if(len(y.unique())==1):
            value,count=np.unique(y, return_counts=True)
            out_put=value[0]
            return TreeNode(None,out_put)
        #having all featuers used up:leaf node
        if(len(features)==0):
            value,count=np.unique(y, return_counts=True)
            y=list(y)
            out_put=max(y,key=y.count) #leaf output
            return TreeNode(None,out_put)
        
        if(level>=self.max_depth):
            value,count=np.unique(y, return_counts=True)
            y=list(y)
            out_put=max(y,key=y.count)
            return TreeNode(None,out_put)
            
        
        max_gain=-math.inf
        best_featindx=None
        for i in  features:
            if(self.criterion=="information_gain"):
                a=X.iloc[:,i]
                curr_gain=information_gain(y,a)
            elif(self.criterion=="gini_index"):
                a=X.iloc[:,i]
                curr_gain=gini_gain(y,a)
            if(curr_gain>max_gain):
                max_gain=curr_gain
                best_featindx=i
        
        value,count=np.unique(y, return_counts=True)
        y=list(y)
        out_put=max(y,key=y.count)       
        #print("type=",type(best_featindx))  
        df=X
        df[df.shape[1]]=y#updated dataframe

        cur_node=TreeNode(best_featindx,out_put)
        #finding all the clases in the feature column
        a=df.iloc[:,best_featindx]
        
        features.remove(best_featindx)
        unique_vals=a.unique()
        for i in unique_vals:
            df1=df[df[best_featindx]==i]
            node=self.grow_tree(df1.iloc[:,0:df1.shape[1]-1],df1.iloc[:,df1.shape[1]-1],features,level+1)
            cur_node.add_child(i,node)
            
        
        return  cur_node
        
    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        feat=[i for i in range(X.shape[1])]

        self.root=self.grow_tree(X,y,features=feat,level=0)
        
            
        pass
        
    def _traverse(self,data_row,node):
        # predicts the class for a given testing point and returns the answer
        
        # We have reached a leaf node
        if len(node.children) == 0 :
            return node.output

        val = data_row[node.data] # represents the value of feature on which the split was made       
        if val not in node.children :
            return node.output
        
        # Recursively call on the splits
        return self._traverse(data_row,node.children[val])


    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        Y = np.array([0 for i in range(len(X))])
        for i in range(len(X)):
            
            Y[i] = self._traverse(X.iloc[i,:],self.root)
        return pd.Series(Y)

        pass
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        if not tree:
            tree = self.root
        
        if len(tree.children) == 0 :
            print("leaf:",tree.output,end="")


        else:
            print(tree.data, "?",end="")
            print(indent, end="")
            for i in (tree.children).values():
               self.print_tree(tree=i,indent= indent + indent)
               print(indent, end="")
        #using self in recursive calling
            


    def plot(self):
        """
        Function to plot the tree
        
        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.print_tree()
        pass

class Node():
    def __init__(self, feature_index=None, compare=None, left=None, right=None, redu_var=None, data=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.compare = compare
        self.left = left
        self.right = right
        self.redu_var = redu_var
        
        # for leaf node
        self.data = data


class DecisionTree2():
    def __init__(self, criterion, max_depth):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.root = None
        
    
        
        self.max_depth = max_depth
        self.criterion=criterion
        
        pass
    def variance_reduction(self, parent, l_child, r_child):
        ''' function to compute variance reduction '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction   

    def best_split(self, df, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the all the necessary values of asplitbest split
        bestsplit = {}
        max_var_red = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = df.iloc[:, feature_index]
            possible_values = np.unique(feature_values)
            # loop over all the feature values present in the data
            for val in possible_values:
                # get current split
                dfleft=df[df[feature_index]<=val]
                dfright=df[df[feature_index]>val]
                
                if len(dfleft)>0 and len(dfright)>0:
                    y, left_y, right_y = df.iloc[:, -1], dfleft.iloc[:, -1], dfright.iloc[:, -1]
                    # compute information gain
                    cur_varredct = self.variance_reduction(y, left_y, right_y)
                    
                    if cur_varredct>max_var_red:
                        bestsplit["featureid"] = feature_index
                        bestsplit["left"] = dfleft
                        bestsplit["compare"]=val
                        bestsplit["right"] = dfright
                        bestsplit["var_red"] = cur_varredct
                        max_var_red = cur_varredct
                        
        # return best split
        return bestsplit      
        
    def grow_tree(self, df, curr_depth=0):
        ''' recursive function to build the tree '''
        
        X=df.iloc[:,:-1]
        Y=df.iloc[:,-1]
    
        n_sam, n_features = np.shape(X)
        best_split = {}
        # split until stopping conditions are met
        if (n_sam>=2 and curr_depth<=self.max_depth):
            # find the best split
            best_split = self.best_split(df, n_sam, n_features)
            # check if information gain is positive
            if best_split["var_red"]>0:
            
                left_subtree = self.grow_tree(best_split["left"], curr_depth+1)
            
                right_subtree = self.grow_tree(best_split["right"], curr_depth+1)
                # return decision node
                return Node(best_split["featureid"], best_split["compare"], 
                            left_subtree, right_subtree, best_split["var_red"])
        
        # compute leaf node
        

        leaf_value = np.mean(Y)  #mode of the class
        # return leaf node
        return Node(data=leaf_value)
    
    
    def fit(self, X, y):
        dataset = X
        dataset[dataset.shape[1]]=y
        self.root = self.grow_tree(dataset,curr_depth=0)
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        pass
    def  traverse(self, x, tree):
        ''' function to predict new dataset '''
        
        if tree.data!=None: 
            return tree.data
        f_val = x[tree.feature_index]
        if f_val<=float(tree.compare):
            return self.traverse(x, tree.left)
        else:
            return self.traverse(x, tree.right)
    
    def predict(self, X):
        Y = np.array([0 for i in range(len(X))])
        for i in range(len(X)):
            Y[i] = self.traverse(X.iloc[i,:],self.root)
        return pd.Series(Y)
        
        #preditions = [self.traverse(x, self.root) for x in X]
        
        return pd.Series(preditions)


    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.data is not None:
            print(tree.data)

        else:
            print("X_"+str(tree.feature_index),"<=", tree.compare, "?")
            print("%sYes:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sNO:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pass

    def plot(self):
        """
        Function to plot the tree
        
        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.print_tree()
        pass

class Node():
    def __init__(self, feature_index=None, compare=None, left=None, right=None, infogain=None, data=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.compare = compare
        self.left = left
        self.right = right
        self.infogain = infogain
        
        # for leaf node
        self.data = data    
class DecisionTree1():
    def __init__(self, criterion, max_depth):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.root = None
        
    
        
        self.max_depth = max_depth
        self.criterion=criterion
        
        pass
    def information_gain(self, p, l, r, mode="gini_index"):
        ''' function to compute information gain '''
        
        weight_l = len(l) / len(p)
        weight_r = len(r) / len(p)
        if mode=="gini_index":
            gain = gini_index(p) - (weight_l*gini_index(l) + weight_r*gini_index(r))
        else:
            gain = entropy(p) - (weight_l*entropy(l) + weight_r*entropy(r))
        return gain
    def best_split(self, df, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the all the necessary values of asplitbest split
        bestsplit = {}
        max_infogain = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = df.iloc[:, feature_index]
            possible_values = np.unique(feature_values)
            # loop over all the feature values present in the data
            for val in possible_values:
                # get current split
                dfleft=df[df[feature_index]<=val]
                dfright=df[df[feature_index]>val]
                
                if len(dfleft)>0 and len(dfright)>0:
                    y, left_y, right_y = df.iloc[:, -1], dfleft.iloc[:, -1], dfright.iloc[:, -1]
                    # compute information gain
                    curr_infogain = self.information_gain(y, left_y, right_y,self.criterion)
                    
                    if curr_infogain>max_infogain:
                        bestsplit["featureid"] = feature_index
                        bestsplit["left"] = dfleft
                        bestsplit["compare"]=val
                        bestsplit["right"] = dfright
                        bestsplit["info_gain"] = curr_infogain
                        max_infogain=curr_infogain
                        
        # return best split
        return bestsplit      
        
    def grow_tree(self, df, curr_depth=0):
        ''' recursive function to build the tree '''
        
        X=df.iloc[:,:-1]
        Y=df.iloc[:,-1]
    
        n_sam, n_features = np.shape(X)
        best_split = {}
        # split until stopping conditions are met
        if (n_sam>=2 and curr_depth<=self.max_depth):
            # find the best split
            best_split = self.best_split(df, n_sam, n_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
            
                left_subtree = self.grow_tree(best_split["left"], curr_depth+1)
            
                right_subtree = self.grow_tree(best_split["right"], curr_depth+1)
                # return decision node
                return Node(best_split["featureid"], best_split["compare"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        Y=list(Y)
        leaf_value = max(Y ,key=Y.count)
        # return leaf node
        return Node(data=leaf_value)
    
    
    def fit(self, X, y):
        dataset = X
        dataset[dataset.shape[1]]=y
        self.root = self.grow_tree(dataset,curr_depth=0)
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        pass
    def  traverse(self, x, tree):
        ''' function to predict new dataset '''
        
        if tree.data!=None: 
            return tree.data
        f_val = x[tree.feature_index]
        if f_val<=float(tree.compare):
            return self.traverse(x, tree.left)
        else:
            return self.traverse(x, tree.right)
    
    def predict(self, X):
        Y = np.array([0 for i in range(len(X))])
        for i in range(len(X)):
            Y[i] = self.traverse(X.iloc[i,:],self.root)
        return pd.Series(Y)
        
        #preditions = [self.traverse(x, self.root) for x in X]
        
        return pd.Series(preditions)


    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.data is not None:
            print(tree.data)

        else:
            print("X_"+str(tree.feature_index),"<=", tree.compare, "?")
            print("%sYes:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sNO:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pass

    def plot(self):
        """
        Function to plot the tree
        
        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.print_tree()
        pass
class DecisionTree4():
    def __init__(self, criterion, max_depth):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """

        pass

    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        pass

    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pass

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
