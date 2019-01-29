

def lowpassfilter(par, k=0.2):
    parF=[]
    for i in range(len(par)):
        if i == 0:
            parF.append(par[i])
        else:
            parF.append(parF[i-1]+k*(par[i]-parF[i-1]))
    return parF


def highpassfilter(par, k=0.2):
    parF=[]
    for i in range(len(par)):
        if i == 0:
            parF.append(par[i])
        else:
            parF.append(par[i]-(parF[i-1]+k*(par[i]-parF[i-1])))
    return parF
