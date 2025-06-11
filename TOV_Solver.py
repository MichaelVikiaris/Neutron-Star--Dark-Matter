# This is a class for solving the TOV equations for a given array-like EoS.
''' The imports should be these
import numpy as np
import math
import matplotlib
from scipy.optimize import fsolve
from scipy.integrate import fixed_quad
from scipy.integrate import solve_ivp
from scipy.differentiate import derivative
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')'''

# The TOV_Solver class. When you create an object you need a filename. After thet, you can soolve for a single configuration using solve_tov and the central pressure, or use the M_R_solve to receive the whole M-R diagram
class TOV_Solver():
    def __init__(self, filename):
        self.pressure = 0
        self.mass = 0
        self.y_tid = 2
        self.lamda = 0
        self.beta = 0
        self.radius = 0
        data = np.genfromtxt(filename,
                     skip_header=0,
                     skip_footer=0,
                     names=True,
                     dtype=None,
                     delimiter='')

        p_b = np.linspace(0,len(data),num=len(data)-1,endpoint=True)
        erho_b = np.linspace(0,len(data),num=len(data)-1,endpoint=True)
        rho_b = np.linspace(0,len(data),num=len(data)-1,endpoint=True)

        for i in range(0,len(data)-1):
            erho_b[i] = data[i][1]
            p_b[i] = data[i][2]
            rho_b[i] = data[i][0]

        self.EoS_b = PchipInterpolator((p_b),(erho_b))#,extrapolate = False)#,kind = 'cubic',bounds_error=False)#,fill_value="extrapolate")
        self.E_n_b = PchipInterpolator((rho_b),(erho_b))
        #f4 = PchipInterpolator(nx1,y1)#,kind = 'cubic',bounds_error=False)#,fill_value="extrapolate")

        self.der_EoS_b = EoS_b.derivative(nu=1)
        self.der_E_n_b = E_n_b.derivative(nu=1)
        
    def dP(self,r,P,M,E):
        DP = -1.474*((E*M)/(r**2))*(1+(P/E))*(1+11.2*(10**(-6))*(r**3)*(P/M))*((1-2.948*(M/r))**(-1))
        return DP

    def dM(self,r,E):
        DM = 11.2*(10**(-6))*(r**2)*E
        return DM

    def EP1(self,P):
        return self.EoS_b(P)

    def PEderom1(self,P):
        return 1/self.der_EoS_b(P)#1/derivative(EP1,P,dx=1e-12)#1e-16

    def F(self,r,E,M,P):
        return (1-1.474*(11.2*10**(-6))*(r**2)*((E)-(P)))*(1-2.948*(M/r))**(-1)

    def Q(self,r,E,M,P,dPdE):
        return 1.474*(11.2*10**(-6))*(r**2)*(5*(E)+9*(P)+((E+P)/(dPdE)))*((1-2.948*(M/r))**(-1))-6*((1-2.948*(M/r))**(-1))-((4*(1.474**2)*M**2)/(r**2))*((1+(11.2*10**(-6))*(r**3)*((P)/M))**2)*((1-2.948*(M/r))**(-2))

    def dy(self,r,y,F,Q):
        return(-y**2-y*F-Q)/r

    def surf_star(self,t,y):
        return y[0]-4*10**(-10)

    surf_star.terminal = True

    # The TOV equations for the one fluid model
    def TOV1(self,r,z):
        E = self.EP1(z[0])
        Fb = self.F(r,E,z[1],z[0])
        dPdE = self.PEderom1(z[0])
        Qb = self.Q(r,E,z[1],z[0],dPdE)
        return [self.dP(r,z[0],z[1],E),self.dM(r,E),self.dy(r,z[2],Fb,Qb)]
    
    def solve_tov(self, pb):
        h = 0.0001
        y10=2
        p0 = pb
        E0 = self.EP1(p0)
        r0 = 0.0001
        m1= (11.2*(10**(-6))/3)*(h**3)*E0
        y0 = [p0,m1,y10]
        sol = solve_ivp(self.TOV1,[0.1,10000],y0,events = self.surf_star)

        mass2 = sol.y[1,len(sol.y[1])-1]
        rad = sol.t[len(sol.t)-1]

        yR = sol.y[2,len(sol.y[2])-1]
        Rs=2.948*mass2
        beta=1.474*(mass2/rad)
        kappa2=(8*beta**5/5)*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*(2*beta*(6-3*yR+3*beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta))**(-1)
        lamda=64/3*kappa2*(rad/Rs)**5#(2/3)*kappa2*(rad/(1.474*mass))
        self.pressure = sol.y[0,len(sol.y[0])-1]
        self.mass = sol.y[1,len(sol.y[1])-1]
        self.y_tid = sol.y[2,len(sol.y[2])-1]
        self.lamda = lamda
        self.beta = beta
        self.radius = rad
