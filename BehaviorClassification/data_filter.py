# File of filters functions. Can be added
def lowpassfilter(par, k=0.2):
    parf=[]
    for i in range(len(par)):
        if i == 0:
            parf.append(par[i])
        else:
            parf.append(parf[i-1]+k*(par[i]-parf[i-1]))
    return parf


def highpassfilter(par, k=0.2):
    parf = []
    for i in range(len(par)):
        if i == 0:
            parf.append(par[i])
        else:
            parf.append(par[i]-(parf[i-1]+k*(par[i]-parf[i-1])))
    return parf

