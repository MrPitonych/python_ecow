
import pandas as pd
import matplotlib.pyplot as plt
import data_filter as flt
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


ActualData = loading('all_info.csv')
ODBA = []
VeDBA = []
SD = []
time = ActualData['time']
Ax = ActualData['gFx']
Ay = ActualData['gFy']
Az = ActualData['gFz']
#paintPar(time, Ax, Ay, Az)
Ax1 = pr.meanpar(Ax, 3500)  # #3500-1 min
SD = pr.sd(Ax, 3500) #
time1 = pr.meantime(time, 3500)
'''Axf = flt.lowpassfilter(Ax, 0.05)    #
Ax2 = pr.meanpar(Axf, 3500)
SD1 = pr.sd(Axf, 3500)'''
'''plt.subplot(221)
plt.plot(time,Ax)  #
plt.title(r'$Ax(x)$')
plt.subplot(222)
plt.plot(time1,SD)
plt.title(r'$Ax - timeW(x)$')
plt.subplot(223)
plt.plot(time,Axf)
plt.title(r'$Ax-F$')
plt.subplot(224)
plt.plot(time1,SD1)
plt.title(r'$Ax-F-timeW$')
plt.show()'''
#paintPar(time, Ax, ODBA, VeDBA)

'''Class = []
g = 9.81
k1 = 0.098
k2 = -0.055
for i in range(len(time1)):
    if VeDBA[i] > (k1*g):
        Class.append(-0.5) #feeding
    elif Ax2[i] > (k2*g):
        Class.append(0)   #standing
    else:
        Class.append(0.5) #lying

mean_feed = Class.count(-0.5)/len(Class)
mean_stand = Class.count(0)/len(Class)
mean_lying = Class.count(0.5)/len(Class)
print(mean_lying, mean_feed, mean_stand )'''
quan= pr.kurtosis(Ax,3500)
plt.plot(time1,quan)
plt.show()