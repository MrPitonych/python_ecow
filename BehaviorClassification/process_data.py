# Test file for all functions
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import data_filter as dtf
import data_parameter as pr
import trait_selection_methods as tsm
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def loading(path):
    df = pd.read_csv(path, usecols=['time','gFx', 'gFy', 'gFz'], sep=';', low_memory=False)
    dictionary = {';':',', ',':'.'}
    df.replace(dictionary, regex=True, inplace=True)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df.dropna(thresh=1)
    df = df.astype(float)
    # df.plot(x='time')
    # plt.show()
    return df


def paintpar(x, par1, par2, par3): #*par
    plt.figure()
    plt.plot(x, par1, "r", x, par2, "g", x, par3, "b")
    plt.legend(['r - 1', 'g - 2', 'b - 3'])
    plt.xlabel("Time")
    plt.ylabel("Parameters")
    plt.show()


def createdataset(time, filename):
    i = 0
    class1 = []
    print("input class: 0 -liying, 1 -standing, 2 - feeding, 3 -noname")
    while i < len(time):
        print("time-interval :", time[i] / 60)
        class1.append(int(input()))
        i += 1
    frame = pd.DataFrame([time, class1])
    frame.to_csv(filename, index=False)
    return frame


ActualData = loading('all_info.csv')
time = ActualData['time']
Ax = ActualData['gFx']
Ay = ActualData['gFy']
Az = ActualData['gFz']

# load dataset
df = pd.read_csv('dataset1.csv', usecols=['Class'], sep=',', low_memory=False)
df = df.astype(int)
Class = df['Class']

np.seterr(all='print')

# Calculate channels parameters
VeDBA = pr.vedba(Ax, Ay, Az)
VeDBA1 = pr.meanpar(VeDBA, 3500)
#activity = pr.activity(Ax, 3500)
Ax = dtf.lowpassfilter(Ax, 0.05)
Ax1 = pr.meanpar(Ax, 3500)  # #3500-1 min
Ay1 = pr.meanpar(Ay, 3500)
#Az1 = pr.meanpar(Az, 3500)
#SD = pr.sd(Ax, 3500)
#quan = pr.quantiles(Ax, 3500, 0.1)
time1 = pr.meantime(time, 3500)
# lda vs pca
#Xr1 = tsm.lda((np.array([Ay1, VeDBA1])).transpose(), Class, [0, 1, 2, 3])
#Xr2 = tsm.pca((np.array([Ay1, VeDBA1])).transpose(), Class, [0, 1, 2, 3])
#plt.show()

#clean noname class
VeDBA1 = pd.Series(VeDBA1).drop(np.where(Class == 3)[0])
Ay1 = pd.Series(Ay1).drop(np.where(Class == 3)[0])
Class = Class.drop(np.where(Class == 3)[0])
# generate test and train data
(train_features, test_features, train_labels, test_labels) = tsm.generatett([VeDBA1, Ay1], Class)


# lda vs pca
#Xr1 = tsm.lda((np.array([Ay1, VeDBA1])).transpose(), Class, [0, 1, 2, 3])
#Xr2 = tsm.pca((np.array([Ay1, VeDBA1])).transpose(), Class, [0, 1, 2, 3])

# Instantiate model with 100 decision trees
y_pred = tsm.randomforest(train_features, test_features, train_labels, 100)

# lda
lda_1 = LinearDiscriminantAnalysis(n_components=2)
x_r = lda_1.fit(train_features, train_labels)#.transform(train_features)
y_pred_lda =lda_1.predict(test_features)

# Сalculation metrics
print("This is Random Forest results :")
print("Accuracy -RF:", metrics.accuracy_score(test_labels, y_pred)) #test_labels
print("Precision- RF:", metrics.precision_score(test_labels, y_pred, average=None))
print("Recall-RF :", metrics.recall_score(test_labels, y_pred, average=None))

print("\n This is Linear Discriminant Analysis results :")
print("Accuracy -LDA:", metrics.accuracy_score(test_labels, y_pred_lda)) #test_labels
print("Precision- LDA:", metrics.precision_score(test_labels, y_pred_lda, average=None))
print("Recall-LDA :", metrics.recall_score(test_labels, y_pred_lda, average=None))

