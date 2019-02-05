"""  Module for filtering data before other operations. It may be used for smoothing graph.
Should be considered that  sensor data already has got  filter process(low pass).
Can  consistently use lowpass filter and highpass filter for create bandpass filter.
"""


def lowpassfilter(param, k=0.2):
    """ Function low pass filter for filtering time series.

    Args:
        param (list): Series for filtering.
        k (float): Filter coefficient. Default 0.2.
    Returns:
        list: The return value. Series after filter.
    """

    parf=[]
    for i in range(len(param)):
        if i == 0:
            parf.append(param[i])
        else:
            parf.append(parf[i-1]+k*(param[i]-parf[i-1]))
    return parf


def highpassfilter(param, k=0.2):
    """ Function high pass filter for filtering time series.

        Args:
            param (list): Series for filtering.
            k (float): Filter coefficient. Default 0.2.
        Returns:
            list: The return value. Series after filter.
        """
    parf = []
    for i in range(len(param)):
        if i == 0:
            parf.append(param[i])
        else:
            parf.append(param[i]-(parf[i-1]+k*(param[i]-parf[i-1])))
    return parf

