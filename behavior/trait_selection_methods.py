# File of selection methods
"""  Module for learning, print and classification. Also the train and test data generate functions are here."""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def lda(X, y, target_names):
    """ Function  for train and print Linear Discriminant Analysis result graph.

            Args:
                X (array): numpy array include two signs.
                y(array): numpy array include all known classes.
            Returns:
                array: The return value. LDA-fit.
        """
    lda_1 = LinearDiscriminantAnalysis(n_components=2)
    x_r = lda_1.fit(X, y).transform(X)
    colors = ['navy', 'turquoise', 'darkorange', 'red']
    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(x_r[y == i, 0], x_r[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.xlabel("first par")
    plt.xlabel("second par")
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of dataset')
    return x_r


def pca(X, y, target_names):
    """ Function  for train and print PCA result graph.

                Args:
                    X (array): numpy array include two signs.
                    y(array): numpy array include all known classes.
                Returns:
                    array: The return value. PCA-fit.
    """
    pca_1 = PCA(n_components=2)
    x_r = pca_1.fit(X).transform(X)
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange', 'red']
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(x_r[y == i, 0], x_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.xlabel("first par")
    plt.xlabel("second par")
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of dataset ')
    return x_r


def randomforest(Xtrain, Xtest, ytrain, ntrees):
    """ Function  for train classification with  Random Forest classifier.

                    Args:
                        Xtrain(array): numpy array include two signs for train.
                        Xtest(array): numpy array include two signs for test.
                        ytrain(array): numpy array include all train classes.
                        ntrees(int): number of trees
                    Returns:
                        array: The return value. Result classifications.
    """
    clf = RandomForestClassifier(n_estimators=ntrees)
    clf.fit(Xtrain, ytrain)
    y_pred = clf.predict(Xtest)
    return y_pred



def generatett(X,y):
    """ Function  for generate test and train data.

                Args:
                    X (array): numpy array include two signs.
                    y(array): numpy array include all known classes.
                Returns:
                   list( array): The return value. Train data, test data, train labels and test labels.
            """
    features = (np.array(X)).transpose()
    train_features, test_features, train_labels,\
    test_labels = train_test_split(features, y, test_size=0.20)
    return (train_features, test_features, train_labels, test_labels)