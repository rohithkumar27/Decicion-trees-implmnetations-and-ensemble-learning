from .base import *
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor, plot_tree

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        if criterion == 'gini':
            self.criterion = criterion
        else:
            self.criterion = 'entropy'

        self.n_estimators = n_estimators
        
        self.estimatorsl = []
        
        self._xs = []
        self._ys = []
        self.y_ht=list()
        
        self.labels = None

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        
        
        
        for estimator in range(self.n_estimators):
            
            #Dtree = DecisionTree(criterion=self.criterion)
            df=X
            df[df.shape[1]]=y
            df2 = df.sample(len(df), replace = True )
            X=df2.iloc[:,:df2.shape[1]-1]
            y=df2.iloc[:,df2.shape[1]-1]
            
            
            Dtree =  DecisionTreeClassifier(max_depth = 4,criterion = self.criterion)

            self._xs.append(X)
            self._ys.append(y)

            Dtree.fit(X,y)
            self.y_ht.append(Dtree.predict(X))

            self.estimatorsl.append(Dtree)

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat =list()
        pred = []
        for i in range(self.n_estimators):
            temp = self.y_ht[i]
            pred.append(temp)

        pred_arr = np.array(pred)
        pred_arr = pred_arr.T
        for i in pred_arr:
            val,count=np.unique(i, return_counts=True)
            y_hat.append( val[np.argmax(count)]) 
        
        
        return(pd.Series(y_hat))
        pass

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        pass



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        
        self.criterion = 'variance'
        

        self.n_estimators = n_estimators
        
        self.estimatorsl = []
        
        self._xs = []
        self._ys = []
        self.y_ht=list()
        
        self.labels = None
        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        for estimator in range(self.n_estimators):
            
            #Dtree = DecisionTree(criterion=self.criterion)
            df=X
            df[df.shape[1]]=y
            df2 = df.sample(len(df), replace = True )
            X=df2.iloc[:,:df2.shape[1]-1]
            y=df2.iloc[:,df2.shape[1]-1]
            
            
            Dtree =  DecisionTreeClassifier(max_depth = 4,criterion = self.criterion)

            self._xs.append(X)
            self._ys.append(y)

            Dtree.fit(X,y)
            self.y_ht.append(Dtree.predict(X))

            self.estimatorsl.append(Dtree)
        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat =list()
        pred = []
        for i in range(self.n_estimators):
            temp = self.y_ht[i]
            pred.append(temp)

        pred_arr = np.array(pred)
        pred_arr = pred_arr.T

        y_hat=np.mean(np.mean(i) for i in pred_arr)
        return pd.Series(y_hat)
        pass

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        pass
