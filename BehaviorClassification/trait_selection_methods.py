from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# lda
def lda(X,y,target_names):
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)
    return X_r2
#pca
#random forest
