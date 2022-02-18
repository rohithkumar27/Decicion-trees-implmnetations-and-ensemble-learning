import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.n_estimators=100
        self.base_estimator=DecisionTreeClassifier()  
        self.trees = []
        self.y_ht=[]

        pass

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        a=np.unique(y)
        self.labels=list(a)
        self.X=X 
        self.y=y 
        
        for n in range(self.n_estimators):
            tree=DecisionTreeClassifier( min_samples_split = 10) 
            
            
            tree.fit(X,y)
            self.y_ht.append(tree.predict(X))

            self.trees.append(tree)
            df=X
            df[df.shape[1]]=y
            df2 = df.sample(len(df), replace = True )
            X=df2.iloc[:,:df2.shape[1]-1]
            y=df2.iloc[:,df2.shape[1]-1]
        
        pass

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point

        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_finalp = list()
        
        pre_vals = {} #class wise count dictionary fro each row

        #for a in range(self.n_estimators):
            #pre_vals[a] = 0
        y_pred=self.y_ht[0]
        y_pred=pd.Series(y_pred)
        a=len(list(y_pred.unique()))
        for j in range(self.n_estimators):
            a=len(list(y_pred.unique()))# predict using each estimator
            cury_hat = self.y_ht[j]
            cury_hat=list(pd.Series(cury_hat).unique())
            if(len(cury_hat)>a):
                y_pred=cury_hat

        y_pred = pd.Series(y_pred)
        return y_pred
    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        pass
