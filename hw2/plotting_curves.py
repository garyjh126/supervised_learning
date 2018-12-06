import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, load_breast_cancer, load_iris, load_digits
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

class Plots():
    # Plotting validation curve 
    def validation_curve(self, classifier, X, y, param_name):
        
        
        param_range = np.logspace(-6, -1, 5)
        train_scores, test_scores = validation_curve(
            classifier, X, y, param_name=param_name, param_range=param_range,
            cv=10, scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Validation Curve with SVM")
        plt.xlabel("$gamma$")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        lw = 2
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                    color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.2,
                        color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                    color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.2,
                        color="navy", lw=lw)
        plt.legend(loc="best")
        plt.show()


    # Plotting learning curve 
    def learning_curve(self, X, y, estimator):

        def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                                n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
            """
            Generate a simple plot of the test and training learning curve.

            Parameters
            ----------
            estimator : object type that implements the "fit" and "predict" methods
                An object of that type which is cloned for each validation.

            title : string
                Title for the chart.

            X : array-like, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.

            y : array-like, shape (n_samples) or (n_samples, n_features), optional
                Target relative to X for classification or regression;
                None for unsupervised learning.

            ylim : tuple, shape (ymin, ymax), optional
                Defines minimum and maximum yvalues plotted.

            cv : int, cross-validation generator or an iterable, optional
                Determines the cross-validation splitting strategy.
                Possible inputs for cv are:
                - None, to use the default 3-fold cross-validation,
                - integer, to specify the number of folds.
                - :term:`CV splitter`,
                - An iterable yielding (train, test) splits as arrays of indices.

                For integer/None inputs, if ``y`` is binary or multiclass,
                :class:`StratifiedKFold` used. If the estimator is not a classifier
                or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

                Refer :ref:`User Guide <cross_validation>` for the various
                cross-validators that can be used here.

            n_jobs : int or None, optional (default=None)
                Number of jobs to run in parallel.
                ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
                for more details.

            train_sizes : array-like, shape (n_ticks,), dtype float or int
                Relative or absolute numbers of training examples that will be used to
                generate the learning curve. If the dtype is float, it is regarded as a
                fraction of the maximum size of the training set (that is determined
                by the selected validation method), i.e. it has to be within (0, 1].
                Otherwise it is interpreted as absolute sizes of the training sets.
                Note that for classification the number of samples usually have to
                be big enough to contain at least one sample from each class.
                (default: np.linspace(0.1, 1.0, 5))
            """
            plt.figure()
            plt.title(title)
            if ylim is not None:
                plt.ylim(*ylim)
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            train_sizes, train_scores, test_scores = learning_curve(
                estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.grid()

            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")

            plt.legend(loc="best")
            return plt


        


        # title = "Learning Curves (Naive Bayes)"
        # # Cross validation with 100 iterations to get smoother mean test and train
        # # score curves, each time with 20% data randomly selected as a validation set.
        # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        # plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

        title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
        # SVC is more expensive so we do a lower number of CV iterations:
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        estimator = SVC(gamma=0.001)
        plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
        
        plt.show()

if __name__ == "__main__":
    p = Plots()
    p.learning_curve()