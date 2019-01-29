
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import data_filter as dtf
import data_parameter as pr
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
    fig = plt.figure()
    plt.plot(x, par1, "r", x, par2, "g", x, par3, "b")
    plt.legend(['r - 1', 'g - 2', 'b - 3'])
    plt.xlabel("Time")
    plt.ylabel("Parameters")
    plt.show()

def createdataset(time, filename):
    i=0
    Class1 = []
    print("input class: 0 -liying, 1 -standing, 2 - feeding, 3 -noname")
    while i < len(time):
        print("time-interval :", time[i] / 60)
        Class1.append(int(input()))
        i += 1
    frame = pd.DataFrame([time, Class1])
    frame.to_csv(filename, index=False)
    return frame

ActualData = loading('all_info.csv')
ODBA = []
VeDBA = []
SD = []
SMA=[]
time = ActualData['time']
Ax = ActualData['gFx']
Ay = ActualData['gFy']
Az = ActualData['gFz']
#paintPar(time, Ax, Ay, Az)
df = pd.read_csv('dataset1.csv', usecols=['Class'], sep=',', low_memory=False)
df = df.astype(int)
Class=df['Class']
np.seterr(all='print')
VeDBA = pr.vedba(Ax,Ay,Az)
VeDBA1 = pr.meanpar(VeDBA,3500)
activity = pr.activity(Ax, 3500)
Ax = dtf.lowpassfilter(Ax,0.05)
Ax1 = pr.meanpar(Ax, 3500)  # #3500-1 min
Ay1 = pr.meanpar(Ay, 3500)
Az1=pr.meanpar(Az,3500)
SD = pr.sd(Ax, 3500) #
quan = pr.quantiles(Ax,3500,0.1)
time1 = pr.meantime(time, 3500)

#lda vs pca
X = np.array([Ay1,VeDBA1])
X = X.transpose()
y = Class
target_names = [0,1,2,3]
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange', 'red']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of dataset 0 -liying, 1 -standing, 2 - feeding')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of dataset 0 -liying, 1 -standing, 2 - feeding')

plt.show()