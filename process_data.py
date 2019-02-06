# Test file for all functions
"""File for test all packets and modules. It's draft"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import behavior.data_filter as dtf
import behavior.data_parameter as pr
import behavior.trait_selection_methods as tsm
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings

warnings.filterwarnings("ignore") # It may be bad


def loading(path):
    df = pd.read_csv(path, usecols=['time','gFx', 'gFy', 'gFz'], sep=';', low_memory=False)
    dictionary = {';': ',', ',': '.'}
    df.replace(dictionary, regex=True, inplace=True)
    print(df)
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


def main(par1, par2, Class, method =None):
    f1_rf=0
    f1_lda=0
    # clean noname class
    par1 = pd.Series(par1).drop(np.where(Class == 3)[0])
    par2 = pd.Series(par2).drop(np.where(Class == 3)[0])
    Class = Class.drop(np.where(Class == 3)[0])
    # generate test and train data
    (train_features, test_features, train_labels, test_labels) = tsm.generatett([par1, par2], Class)

    if method != "LDA" and method != "RF" and method != None:
        print("This isn't correct a little")
        method = None
    if method =="RF" or method == None:
        # Instantiate model with 100 decision trees
        y_pred = tsm.randomforest(train_features, test_features, train_labels, 100)
        # Сalculation metrics
        #print("Test Class :", train_labels)
        #print("\nThis is Random Forest results :")
        #print("Precision- RF:", metrics.precision_score(test_labels, y_pred, average=None))
        #print("Recall-RF :", metrics.recall_score(test_labels, y_pred, average=None))
        f1_rf = metrics.f1_score(test_labels, y_pred, average='macro')
        #print("F1-RF :", f1_rf)

    if method == "LDA" or method == None:
        # display with lda
        #tsm.lda((np.array([par1, par2])).transpose(), Class, ["Lying", "Standing", "Feeding"])
        #plt.show()
        # lda
        lda_1 = LinearDiscriminantAnalysis(n_components=2)
        x_r = lda_1.fit(train_features, train_labels)  # .transform(train_features)
        y_pred_lda = lda_1.predict(test_features)
        #print("\n This is Linear Discriminant Analysis results :")
        #print("Precision- LDA:", metrics.precision_score(test_labels, y_pred_lda, average=None))
        #print("Recall-LDA :", metrics.recall_score(test_labels, y_pred_lda, average=None))
        f1_lda =metrics.f1_score(test_labels, y_pred_lda, average='macro')
        #print("F1 -LDA:",f1_lda )
    return (f1_rf,f1_lda)

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
#VeDBA = pr.vedba(Ax, Ay, Az)
#VeDBA1 = pr.meanpar(VeDBA, 3500)
activity = pr.activity(Ax, 3500)
Ax = dtf.lowpassfilter(Ax, 0.05)
Ax1 = pr.meanpar(Ax, 3500)  # #3500-1 min
Ay1 = pr.meanpar(Ay, 3500)
Az1 = pr.meanpar(Az, 3500)
#SD = pr.sd(Ax, 3500)
#SDy = pr.sd(Ay, 3500)
#quan = pr.quantiles(Ax, 3500, 0.1)

time1 = pr.meantime(time, 3500)
print(time1)
d={'time':time1,'gFx': Ax1,'gFy': Ay1, 'gFz':Az1}
frame = pd.DataFrame(data =d)
frame.to_csv('new_data2.csv', index=False)

print(help(pr.meanpar))
#It's for exam methods
'''
print("With Ay1, VeDBA1")
f1 = [],[]
for i in range(100):
   (f1_rf, f1_lda) = main(Ay1, VeDBA1, Class)
   f1[0].append(f1_rf)
   f1[1].append(f1_lda)

print("RF-median: ", float(np.median(f1[0])), "LDA-median: ", float(np.median(f1[1])))
print("RF-min: ", float(np.min(f1[0])), "LDA-min: ", float(np.min(f1[1])))

print("With VeDBA1, Az1")
f1 = [],[]
for i in range(100):
   (f1_rf, f1_lda) = main(VeDBA1, Az1, Class)
   f1[0].append(f1_rf)
   f1[1].append(f1_lda)

print("RF-median: ", float(np.median(f1[0])), "LDA-median: ", float(np.median(f1[1])))
print("RF-min: ", float(np.min(f1[0])), "LDA-min: ", float(np.min(f1[1])))

print("With VeDBA1, activity")
f1 = [],[]
for i in range(100):
   (f1_rf, f1_lda) = main(VeDBA1,activity, Class)
   f1[0].append(f1_rf)
   f1[1].append(f1_lda)

print("RF-median: ", float(np.median(f1[0])), "LDA-median: ", float(np.median(f1[1])))
print("RF-min: ", float(np.min(f1[0])), "LDA-min: ", float(np.min(f1[1])))
print("With VeDBA1, Ax1")
f1 = [],[]
for i in range(100):
   (f1_rf, f1_lda) = main(VeDBA1, Ax1, Class)
   f1[0].append(f1_rf)
   f1[1].append(f1_lda)

print("RF-median: ", float(np.median(f1[0])), "LDA-median: ", float(np.median(f1[1])))
print("RF-min: ", float(np.min(f1[0])), "LDA-min: ", float(np.min(f1[1])))

print("With VeDBA1, SD")
f1 = [],[]
for i in range(100):
   (f1_rf, f1_lda) = main(VeDBA1, SD, Class)
   f1[0].append(f1_rf)
   f1[1].append(f1_lda)

print("RF-median: ", float(np.median(f1[0])), "LDA-median: ", float(np.median(f1[1])))
print("RF-min: ", float(np.min(f1[0])), "LDA-min: ", float(np.min(f1[1])))

print("With VeDBA1, SDy")
f1 = [],[]
for i in range(100):
   (f1_rf, f1_lda) = main(VeDBA1, SDy, Class)
   f1[0].append(f1_rf)
   f1[1].append(f1_lda)

print("RF-median: ", float(np.median(f1[0])), "LDA-median: ", float(np.median(f1[1])))
print("RF-min: ", float(np.min(f1[0])), "LDA-min: ", float(np.min(f1[1])))

print("With VeDBA1, quan")
f1 = [],[]
for i in range(100):
   (f1_rf, f1_lda) = main(VeDBA1, quan, Class)
   f1[0].append(f1_rf)
   f1[1].append(f1_lda)

print("RF-median: ", float(np.median(f1[0])), "LDA-median: ", float(np.median(f1[1])))
print("RF-min: ", float(np.min(f1[0])), "LDA-min: ", float(np.min(f1[1])))
'''