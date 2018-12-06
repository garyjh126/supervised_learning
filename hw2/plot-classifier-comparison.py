

"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, load_breast_cancer, load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
import pandas as pd
import plotting_curves as pc

test_size = 40000
train_data_scope = 100000
test_data_scope = 10000


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


def getXy(plotgraphs = False, plot_samples = 0):
   
    filename = "./train/train.csv"
    df = pd.read_csv(filename, names=['1','2','3','4','5','6','7','8','target'])
    features = ['1','2','3','4','5','6','7','8']
    
    dftoplot = pd.read_csv(filename, names=['1','2','3','4','5','6','7','8','target'], nrows=plot_samples)
    
    
    if plotgraphs == True:
        plt.scatter(dftoplot[['4']],dftoplot[['7']],marker='+',color='red')
        plt.plot()
        plt.show()

    X = df.loc[:train_data_scope-1, features].values
    y = df.loc[:train_data_scope-1,['target']].values
    X = np.reshape(X, (train_data_scope, 8))
    y = np.reshape(y, train_data_scope)
    return X, y

### ML tutorial 

# # Load dataset
# data = load_breast_cancer()

# # Organize data for testing with breast cancer dataset
# label_names = data['target_names']
# labels = data['target']
# feature_names = data['feature_names']
# features = data['data']


# Loading in our own data
X, y = getXy(plotgraphs = True, plot_samples=200)

# Split our data
X, test_X, y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)

# FEATURE SCALING 

# Feature scaling is the method to limit the range of variables so that they can be compared on common grounds. It is performed on
# continuous variables. Lets plot the distribution of all the continuous variables in the data set.
# Standardization is a useful technique to transform attributes with a Gaussian distribution and differing means and standard 
# deviations to a standard Gaussian distribution with a mean of 0 and a standard deviation of 1.

scaler = StandardScaler().fit(X)
#rescaledX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)



## Naive Bayes (NB)

# Initialize our classifier
gnb = GaussianNB()


# Train our classifier
model = gnb.fit(X, y)

# Make predictions
preds = gnb.predict(test_X)


# Evaluate accuracy
print("Using naive bayes\nAccuracy: " , accuracy_score(test_y, preds)) # As you see in the output, the NB classifier is 94.15% accurate on the breast cancer dataset. 
                                          # These results suggest that our feature set of 30 attributes are good indicators of tumor class. 

# When running the model on the breast cancer dataset, we get a 52% accuracy. 
# Diagnosing bias vs variance (Underfitting or Overfitting).
# The bias of an estimator is its average error for different training sets. The variance of an estimator indicates how sensitive it 
# is to varying training sets. Noise is a property of the data.


def print_test():
    print(preds, test_y)
    print(len(preds) - 32500, len(test_y) - 32500)
    count_correct = 0
    for i in range(len(preds) - 32800):
        print("{}: {},{}".format(i, preds[i], test_y[i]))  
        if preds[i] == test_y[i]:
            count_correct+=1
    print(count_correct)




## MLP Classifier

# Initialize our classifier
mlp = MLPClassifier(alpha=1)


# Train our classifier
model = mlp.fit(X, y)

# Make predictions
preds = mlp.predict(test_X)

# Evaluate accuracy
print("Using MLPClassifier\nAccuracy: " , accuracy_score(test_y, preds))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')


X_train_std = scaler.transform(X)
X_test_std = scaler.transform(test_X)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y, test_y))
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=mlp, test_idx=range(105, 150))
plt.xlabel('X [standardized]')
plt.ylabel('y [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_01.png', dpi=300)
plt.show()

from matplotlib.colors import ListedColormap
