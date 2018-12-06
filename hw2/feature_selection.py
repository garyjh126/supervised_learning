
from sklearn.base import clone
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

train_data_scope = 100000
test_data_scope = 10000

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

X, y = getXy()
print(X, y)

X_train, X_test, y_train, y_test =  train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)



# # Bringing features onto the same scale


from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)



from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

mlp = MLPClassifier(alpha=1)


# selecting features
sbs = SBS(mlp, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.1, 1.89])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
# plt.savefig('images/04_08.png', dpi=300)
plt.show()



# Optimal no. of features seems to be 6 features

