import random
from random import sample
import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier
class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        if(base_estimator==None):
            base_estimator=DecisionTreeClassifier(max_depth=1)
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.estimatorsl=list()
        self.alphas=[]
        self.y_ht=[]
        pass

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        a=np.unique(y)
        self.labels=list(a)
        for estimator in range(self.n_estimators):
            n_samples=X.shape[0]

            sam_weights=[1/n_samples]*n_samples

            tree=DecisionTreeClassifier(max_depth=1,criterion='entropy')
            tree.fit(X,y,sample_weight=sam_weights)
            self.y_ht.append(tree.predict(X))
            y_h=tree.predict(X)

            self.estimatorsl.append(tree)

            wrong_p=0.0000001

            wrong_plist=list()

            for i in range(len(y)):
                if(y_h[i]!=y[i]):
                    wrong_p+=sam_weights[i]
                    wrong_plist.append(i)

            
            alpha=0.5*(math.log2(((1-wrong_p)/wrong_p)))
            self.alphas.append(alpha)

            #updated weights
            for i in range(len(y)):
                if(i in wrong_plist):
                    sam_weights[i]=sam_weights[i]*math.exp(alpha)
                else:
                     sam_weights[i]=sam_weights[i]*math.exp(-alpha)   

            sam_weights=[w/sum(sam_weights) for w in sam_weights]
            df=X
            df[df.shape[1]]=y
            df2 = df.sample(len(df), replace = True, weights =sam_weights )
            X_new=df2.iloc[:,:df2.shape[1]-1]
            y_new=df2.iloc[:,df2.shape[1]-1]
        pass

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_finalp = list()
        
        pre_vals = {} #class wise alpha  performance dictionary fro each row

        for a in self.labels:
            pre_vals[a] = 0

        for i in range(len(X)):
            
            for j in range(self.n_estimators):
                # predict using each estimator
                cury_hat = self.y_ht[j]
                cury_hat=list(cury_hat)

                # aadding the alphas on each label bucket based on the each estimator for_ each row
                pre_vals[cury_hat[i]] += self.alphas[j]


                # pred_values[self.estimators_list[j].predict(curr_sample)] += self.all_amount_of_says[j]
            
            # get the class with max alpha
            pre_max_class = max(pre_vals, key= lambda x: pre_vals[x])
            
            y_finalp.append(pre_max_class)
        
        y_hat = pd.Series(y_finalp)
        return y_hat



        pass

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """






        pass
