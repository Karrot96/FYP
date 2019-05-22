import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

def func(x, a, b, c):
    return (a**3 * b * x**5 -a*b*x**3)


xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')

popt, pcov = curve_fit(func, xdata, ydata)

plt.plot(xdata,
         func(xdata, *popt),
         'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt)
)
plt.show()