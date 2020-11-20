import numpy as np 

d = np.loadtxt(r'G:\Subject\1.1\2.1\cs_y.csv',skiprows=1,delimiter=',',usecols=(0),unpack=True)
for i in range(len(d)):
    if  d[i] < 0.1:
        d[i] = 1
    elif 0.1 <= d[i] and d[i] < 0.25:
        d[i] = 2
    elif 0.25 <= d[i] and d[i] < 0.4:
        d[i] = 3
    elif 0.4 <= d[i] and d[i] < 1:
        d[i] = 4
    else:
        d[i] =5
d = d.astype(np.int8)
np.savetxt(r'G:\Subject\1.1\2.1\plot\re_label.txt',d,fmt='%i')

y = np.loadtxt(r'G:\Subject\1.1\2.1\y.csv',skiprows=1,delimiter=',',usecols=(1),unpack=True)
for i in range(len(y)):
    if  y[i] < 0.1:
        y[i] = 1
    elif 0.1 <= y[i] and y[i] < 0.25:
        y[i] = 2
    elif 0.25 <= y[i] and y[i] < 0.4:
        y[i] = 3
    elif 0.4 <= y[i] and y[i] < 1:
        y[i] = 4
    else:
        y[i] =5
y = y.astype(np.int8)
np.savetxt(r'G:\Subject\1.1\2.1\plot\pr_label.txt',y,fmt='%i')