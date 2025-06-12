#Code to create the EoS for neutron and dark matter. For neutron matter, we need the array-like EoS in the form of rho|e_rho|p_b. 
# The imports should be these
import numpy as np
import math
import matplotlib
from scipy.optimize import fsolve
from scipy.integrate import fixed_quad
from scipy.integrate import solve_ivp
from scipy.differentiate import derivative
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt # The matplotlib ONLY in case you need to test it and see the results
matplotlib.use('TkAgg')

# EoS for the Ordinary Matter
class NS_EoS():
    def __init__(self,filename):
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

        self.der_EoS_b = self.EoS_b.derivative(nu=1)
        self.der_E_n_b = self.E_n_b.derivative(nu=1)

# Relativistic DM EoS
class DM_EoS_rel():
    def __init__(self,z,mx):
        hc=197.327
        rows=10000
        cols=5
        data = np.zeros((rows,cols))

        nxmat = np.geomspace(2*10**(-7),10**(2),10000,endpoint=True)

        data[:,0] = nxmat

        self.z = z
        self.mx = mx

        def kfx(nx):
            return (hc * (3 * np.pi**2 * nx)**(1/3))

        def x(nx):
             return (hc * (3 * np.pi**2 * nx)**(1/3)) / mx

        def ex(nx):
            e1 = mx**4/(hc**3*8*np.pi**2)
            e2 = x(nx)*np.sqrt(1+x(nx)**2)*(1+2*x(nx)**2)-np.log(x(nx)+np.sqrt(1+x(nx)**2))
            e3 = (nx**2*hc**3)/(2*z**2)
            return e1*e2+e3

        def der_e_x(nx): # The derivative of the Energy Density
            #return derivative(e_x, n_x, initial_step = 0.0001).df
            return np.sqrt(kfx(nx)**2 + mx**2) + 2 * nx * hc**3 / (2 * z**2)

        def Px(nx): # The pressure of Dark Matter
            return nx * der_e_x(nx) - ex(nx)
            #return derivative(ex,nx).df

        def PEderdm(ni):
            e = ex(ni)
            return derivative(Px,e)

        chemx = np.zeros((rows))
        exmat = np.zeros((rows))
        pxmat = np.zeros((rows))
        dpdemat = np.zeros((rows))

        for l in range(0,rows):
            valex = ex(nxmat[l])
            exmat[l] = valex
            valpx = Px(nxmat[l])
            pxmat[l] = valpx
            valdpde = PEderdm(nxmat[l])
            dpdemat[l] = valdpde.df

        data[:,1] = chemx[:]
        data[:,2] = exmat[:]
        data[:,3] = pxmat[:]
        data[:,4] = dpdemat[:]

        x1=np.linspace(0,len(data),num=len(data)-1,endpoint=True)
        y1=np.linspace(0,len(data),num=len(data)-1,endpoint=True)
        nx1=np.linspace(0,len(data),num=len(data)-1,endpoint=True)

        for l in range(0,len(data)-1):
            y1[l] = data[l][2]
            x1[l] = data[l][3]
            nx1[l] = data[l][0]

        self.EoS_x = PchipInterpolator(x1,y1)#interp1d(x1,y1,kind = 'cubic',bounds_error=False)#,fill_value="extrapolate")
        self.E_n_x = PchipInterpolator(y1,nx1)#interp1d(y1,nx1,kind = 'cubic',bounds_error=False)#,fill_value="extrapolate")
        self.der_EoS_x = self.EoS_x.derivative(nu=1)
        self.der_E_n_x = self.E_n_x.derivative(nu=1)

# This is the code for the non-relativistic DM EoS. Just in case that the particle is heavy and non-relativistic
class DM_EoS_nonrel():
    def __init__(self,z,mx):
        hc=197.327
        rows=10000
        cols=5
        data = np.zeros((rows,cols))

        nxmat = np.geomspace(2*10**(-7),10**(2),10000,endpoint=True)

        data[:,0] = nxmat

        self.z = z
        self.mx = mx

        def kfx(nx):
            return (hc * (3 * np.pi**2 * nx)**(1/3))

        def x(nx):
             return (hc * (3 * np.pi**2 * nx)**(1/3)) / mx

        def ex(nx):
            e1 = mx*nx
            e2 = (hc**2*(3*np.pi**2*nx)**(5/3))/(10*np.pi**2*mx)
            e3 = (nx**2*hc**3)/(2*z**2)
            return e1+e2+e3

        def der_e_x(nx): # The derivative of the Energy Density
            #return derivative(e_x, n_x, initial_step = 0.0001).df
            return np.sqrt(kfx(nx)**2 + mx**2) + 2 * nx * hc**3 / (2 * z**2)

        def Px(nx): # The pressure of Dark Matter
            return nx * der_e_x(nx) - ex(nx)
            #return derivative(ex,nx).df

        def PEderdm(ni):
            e = ex(ni)
            return derivative(Px,e)

        chemx = np.zeros((rows))
        exmat = np.zeros((rows))
        pxmat = np.zeros((rows))
        dpdemat = np.zeros((rows))

        for l in range(0,rows):
            valex = ex(nxmat[l])
            exmat[l] = valex
            valpx = Px(nxmat[l])
            pxmat[l] = valpx
            valdpde = PEderdm(nxmat[l])
            dpdemat[l] = valdpde.df

        data[:,1] = chemx[:]
        data[:,2] = exmat[:]
        data[:,3] = pxmat[:]
        data[:,4] = dpdemat[:]

        x1=np.linspace(0,len(data),num=len(data)-1,endpoint=True)
        y1=np.linspace(0,len(data),num=len(data)-1,endpoint=True)
        nx1=np.linspace(0,len(data),num=len(data)-1,endpoint=True)

        for l in range(0,len(data)-1):
            y1[l] = data[l][2]
            x1[l] = data[l][3]
            nx1[l] = data[l][0]

        self.EoS_x = PchipInterpolator(x1,y1)#interp1d(x1,y1,kind = 'cubic',bounds_error=False)#,fill_value="extrapolate")
        self.E_n_x = PchipInterpolator(y1,nx1)#interp1d(y1,nx1,kind = 'cubic',bounds_error=False)#,fill_value="extrapolate")
        self.der_EoS_x = self.EoS_x.derivative(nu=1)
        self.der_E_n_x = self.E_n_x.derivative(nu=1)
