import numpy as np
import re
import pandas as pd
import csv

class AdalineGD(object):
    
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.training_samples = ''
   
    def __str__(self):
        return self.training_samples

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        print(len(self.training_samples))
        self.w_ = np.random.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,  
            # in the case of logistic regression (as we will see later), 
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

    def read_training_data(self):
        with open('train/train.txt') as f:
            f_csv = csv.reader(f)
            # _, *training_data, label, _ = re.split(r'[\s]', row)
            data = list(f_csv)
           
            df = pd.DataFrame(data)
            print(df.shape)
            
            training_labels = data[1][3]

            for row in data:
                for cell in row:
                    
            #print(df)
            print(data[0], data[1])
            # for row in range(10):
                    
            #     for num in range():
            #         #if(num!='' or num!='0' or num!='1'):
            #         # if(len(str(x[num]))>3):
                        
            #         #     value2=(str(x[num])).replace(',', '.')
            #         #     x[num] = float(value2)
            #         print(num)

            
            #self.training_samples = np.array(x).astype('str')

            #self.training_samples.append(data)
           
       
        return (self.training_samples)

        
if __name__ == "__main__":
    agd = AdalineGD()
    data = agd.read_training_data()
    # training_samples = data[0:, 1:9]
    # training_labels = data[0:, 9:10]
    
    #agd.fit(training_samples, training_labels)
    #agd.fit(agd.training_samples, agd.training_samples)