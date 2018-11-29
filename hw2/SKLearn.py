import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn import svm
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split

test_size = 4000
data_scope = 10000

# Loading the data 

filename = "./train/train.csv"

df = pd.read_csv(filename, names=['1','2','3','4','5','6','7','8','target'])

features = ['1','2','3','4','5','6','7','8']

# Separating out the features
X = df.loc[:data_scope, features].values

# Separating out the target
y = df.loc[:data_scope,['target']].values




# Standardizing the features
X = StandardScaler().fit_transform(X)
Standardize_DF = pd.DataFrame(X)

X = np.reshape(X, (data_scope+1, 8))
y = np.reshape(y, data_scope+1)

# Just dividing train.csv such that 40% of data is contained in X_test and y_test variables
X, X_test, y, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

# Scikit Learn Machine Learning SVM
clf = svm.SVC(kernel = 'linear',  C=1.0)
clf.fit(X, y)
print("SVM Accuracy: ", clf.score(X_test, y_test))



def use_testing_data(self):
    
    print("\n----------------------\nPerformance on test data:\n")
    testdata_scope = 10000-test_size

    # Loading the test data 

    filename = "./test/test.csv"
    test_df = pd.read_csv(filename, names=['1','2','3','4','5','6','7','8'])
    test_features = ['1','2','3','4','5','6','7','8']

    # Separating out the features
    test_X = test_df.loc[:testdata_scope, test_features].values


    # Standardizing the features
    # test_X = StandardScaler().fit_transform(test_X)
    # Standardize_DF = pd.DataFrame(test_X)

    test_X = np.reshape(test_X, (testdata_scope+1, 8))
    y = np.reshape(y, testdata_scope+1)



    # Scikit Learn Machine Learning SVM
    clf = svm.SVC(kernel = 'linear',  C=1.0)
    clf.fit(test_X, y)
    print("Accuracy: ", clf.score(test_X, y))

    myfile = open('xyz.txt', 'w')
    for line_no in range(len(test_X)):
        #print('Prediction: ', test_X[line_no], clf.predict(test_X[line_no].reshape(1,-1)))
        myfile.write("{} .. Prediction: {}\n".format(test_X[line_no], clf.predict(test_X[line_no].reshape(1,-1))))

    myfile.close()


    #print('Prediction: ', clf.predict(test_X[-1].reshape(1,-1)))


