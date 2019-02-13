import numpy as np


def jerkFilter(x, y, z, dt):
    jerk_x = x[dt:len(x)] - x[0:(len(x) - dt)]
    jerk_y = y[dt:len(y)] - y[0:(len(y) - dt)]
    jerk_z = z[dt:len(z)] - z[0:(len(z) - dt)]

    g = np.zeros(len(x) * 3).reshape(len(x), 3)  # matrix

    for i in range(len(x)):
        g[i, 0] = np.mean(x)
        g[i, 1] = np.mean(y)
        g[i, 2] = np.mean(z)
        x = x - g[i, 0]
        y = y - g[i, 1]
        z = z - g[i, 2]

    cos_alpha = (x[dt:len(x)] * x[0:(len(x) - dt)] + y[dt:len(x)] * y[0:(len(y) - dt)] +
                 z[dt:len(z)] * z[0:(len(z) - dt)]) / (np.sqrt(
        x[dt:len(x)] * x[dt:len(x)] + y[dt:len(y)] * y[dt:len(y)] +
        z[dt:len(z)] * z[dt:len(z)])) / (np.sqrt(
        x[0:(len(x) - dt)] * x[0:(len(x) - dt)] + y[0:(len(y) - dt)] * y[0:(len(y) - dt)] +
        z[0:(len(z) - dt)] * z[0:(len(z) - dt)]))
    cos_alpha[cos_alpha > 1] = 1
    cos_alpha[cos_alpha < -1] = -1

    alpha = np.arccos(cos_alpha) * 180 / 3.14
    a = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))
    c = np.sqrt((jerk_x ** 2) + (jerk_y ** 2) + (jerk_z ** 2))

    sj = c

    for i in range(dt, len(x)):
        if a[i] < a[i - dt]:
            c[i - dt] = -c[i - dt]
        sj[i - dt] = (1 + alpha[i - dt] / 180) * c[i - dt]
    return sj
