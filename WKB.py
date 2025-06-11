import numpy as np
import math
import matplotlib.pyplot as plt
a = 10
b = 0.1
hbar = 1973.3
m0 = 5.11*(10**5)
def Vx(x):
    return b*np.sqrt(1-(x**2/(a**2))) #Insert the potential here 
def px(V,E,m):
    return np.sqrt((V-E))
def lamda(I):
    return np . exp (-0.5123094089*I)
def Ptrans(l):
    return (4/(abs(2/l+l/2)**2))
Ptransmat = np.array([])
Emat = np.array([])
r = np.linspace(-a, a, 5000)
Vmat = Vx(r)
plt.plot(r, Vmat, "r", label = "V potential")
plt.legend(loc = "best")
plt.show ( )
for c in range (1, 1000):
    E0 = 0.0001 * c
    Emat = np.append(Emat, E0)
    Vmat = Vx(r)
    k=0
    while E0 >= Vmat[k]:
        k = k + 1
        indx = k
    tpE = Vmat[k]
    tp = -r[k]
    x1 = 0
    lmat = np.array([])
    h = tp/10
    m0 = 1 #9.1∗10∗∗(−31)
    j = 2
    V0 = Vx(x1)
    p0 = px(V0, E0, m0)
    f0=abs(p0)
    ol = h / 3.0 * (f0)
    for i in range (1,10):
        x1 = x1 + h
        V0 = Vx(x1)
        p0 = px(V0,E0,m0)
        if j == 4:
            j = 2
        elif j == 2:
            j = 4
        f0 = abs(p0)
        ol = ol + h / 3.0 * (j * f0)
    x1 = x1 + h
    V0 = Vx(x1)
    p0 = px(V0,E0,m0)
    f0 = abs(p0)
    ol = ol + h /3.0 * f0
    lamda0 = lamda(ol + ol)
    Ptrans0 = Ptrans(lamda0)
    Vmat = Vx(r)
    Ptransmat = np.append(Ptransmat, Ptrans0)
    print(c)
plt.plot(Emat, Ptransmat, "r" , label = "V potential")
plt.xlabel("Energy")
plt.ylabel("Ptrans")
plt.legend(loc = "best")
plt.show()
