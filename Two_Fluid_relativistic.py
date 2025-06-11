''' THE IMPORTS AND THE IMPORTANT VARIABLES FOR THE CODE'''
import numpy as np

import math

from scipy.optimize import fsolve
from scipy.integrate import fixed_quad
from scipy.integrate import solve_ivp
from scipy.differentiate import derivative

from scipy.interpolate import PchipInterpolator

#matplotlib.use('nbagg')
matplotlib.use('TkAgg')  # Use the TkAgg backend for GUI rendering
import matplotlib.pyplot as plt

c = 2.9979*10**10 
G = 6.67408*10**(-8) 
Msun = 1.989*10**33 
Length = G*Msun/c**2 
Time = Length/c 
MeV = 1.60217662 * 10**(-13)  
e0 = MeV*10**7/10**(-39)  
Density = Msun/Length**3  
transform = e0/(Density*c**2) 

data = np.genfromtxt('NSEOS-BonnA-NES.dat', # Import your EoS in the form of an array
                     skip_header=0,
                     skip_footer=0,
                     names=True,
                     dtype=None,
                     delimiter='')


x1=np.linspace(0,len(data),num=len(data)-1,endpoint=True)
y1=np.linspace(0,len(data),num=len(data)-1,endpoint=True)
nx1=np.linspace(0,len(data),num=len(data)-1,endpoint=True)

for i in range(0,len(data)-1):
    y1[i] = data[i][1]
    x1[i] = data[i][2]
    nx1[i] = data[i][0]

f3 = PchipInterpolator(x1,y1)
f4 = PchipInterpolator(y1,nx1)

derivativef3 = f3.derivative(nu=1)

#-------------------- RELATIVISTIC DARK MATTER EOS -----------------------

hc=197.327
rows=10000
cols=5
data = np.zeros((rows,cols))

nxmat = np.geomspace(2*10**(-7),10**(2),10000,endpoint=True) # The values of n_x start from 2*10**(-7) because otherwise you wont have good interpolation values

data[:,0] = nxmat

# You can insert the desired values of interaction (z) and dm particle mass (mx)
z=2
mx = 1500

def kfx(nx):
    return (hc * (3 * np.pi**2 * nx)**(1/3))

def x(nx):
     return (hc * (3 * np.pi**2 * nx)**(1/3)) / mx

def ex(nx):
    e1 = mx**4/(hc**3*8*np.pi**2)
    e2 = x(nx)*np.sqrt(1+x(nx)**2)*(1+2*x(nx)**2)-np.log(x(nx)+np.sqrt(1+x(nx)**2))
    e3 = (nx**2*hc**3)/(2*z**2)
    return e1*e2+e3

'''
In case of a very heavy particle, one can also use the non-relativistic EoS for DM. To do that, you simply change the e_x equation, with something a little bit simpler
def ex(nx):
    e1 = mx*nx
    e2 = (hc**2*(3*np.pi**2*nx)**(5/3))/(10*np.pi**2*mx)
    e3 = (nx**2*hc**3)/(2*z**2)
    return e1+e2+e3

Then you just follow the standard procedure since P_x is n_x*mu_x - e_x and mu_x is the derivative of e_x in terms of n_x
'''

def der_e_x(nx): # The derivative of the Energy Density
    #return derivative(e_x, n_x, initial_step = 0.0001).df
    return np.sqrt(kfx(nx)**2 + mx**2) + 2 * nx * hc**3 / (2 * z**2)

def Px(nx): # The pressure of Dark Matter
    return nx * der_e_x(nx) - ex(nx)
    #return derivative(ex,nx).df

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

f5 = PchipInterpolator(x1,y1)
f6 = PchipInterpolator(y1,nx1)

derivativef5 = f5.derivative(nu=1)

# ------------------------- Define your equations needed for the TOV system -----------------------------

def dP(r,P,M,E): # Pressure
    DP = -1.474*((E*M)/(r**2))*(1+(P/E))*(1+11.2*(10**(-6))*(r**3)*(P/M))*((1-2.948*(M/r))**(-1))
    return DP
def dM(r,E): # Mass
    DM = 11.2*(10**(-6))*(r**2)*E
    return DM

def dPom(r,Pom,Pdm,M,Eom): # Ordinary matter pressure
    P=Pom+Pdm
    DPOM = -1.474*((Eom*M)/(r**2))*(1+(Pom/Eom))*(1+11.2*(10**(-6))*(r**3)*(P/M))*((1-2.948*(M/r))**(-1))
    return DPOM

def dPdm(r,Pom,Pdm,M,Edm): # Dark matter pressure
    P=Pom+Pdm
    DPDM = -1.474*((Edm*M)/(r**2))*(1+(Pdm/Edm))*(1+11.2*(10**(-6))*(r**3)*(P/M))*((1-2.948*(M/r))**(-1))
    return DPDM

def dMom(r,Eom): # Ordinary matter mass
    DMOM = 11.2*(10**(-6))*(r**2)*Eom
    return DMOM

def dMdm(r,Edm): # Dark matter mass
    DMDM = 11.2*(10**(-6))*(r**2)*Edm
    return DMDM

def EP1(P): # Ordinary matter EoS
    return f3(P)
    
def ex(P): # Dark matter EoS
    return f5(P)

# The following F and Q components are needed for the Tidal Polarizability
def Fr(r,Eom,Edm,M,Pom,Pdm):
    return (1-1.474*(11.2*10**(-6))*(r**2)*((Eom+Edm)-(Pom+Pdm)))*(1-2.948*(M/r))**(-1)

def rQ(r,Eom,M,Pom,Edm,Pdm,dPdEom,dPdEdm):
    return 1.474*(11.2*10**(-6))*(r**2)*(5*(Eom+Edm)+9*(Pom+Pdm)+((Eom+Pom)/(dPdEom)+(Edm+Pdm)/(dPdEdm)))*((1-2.948*(M/r))**(-1))-6*((1-2.948*(M/r))**(-1))-((4*(1.474**2)*M**2)/(r**2))*((1+(11.2*10**(-6))*(r**3)*((Pom+Pdm)/M))**2)*((1-2.948*(M/r))**(-2))

def F(r,E,M,P):
    return (1-1.474*(11.2*10**(-6))*(r**2)*((E)-(P)))*(1-2.948*(M/r))**(-1)

def Q(r,E,M,P,dPdE):
    return 1.474*(11.2*10**(-6))*(r**2)*(5*(E)+9*(P)+((E+P)/(dPdE)))*((1-2.948*(M/r))**(-1))-6*((1-2.948*(M/r))**(-1))-((4*(1.474**2)*M**2)/(r**2))*((1+(11.2*10**(-6))*(r**3)*((P)/M))**2)*((1-2.948*(M/r))**(-2))

def PEderom1(P): # Speed of sound of Ordinary Matter
    return 1/derivativef3(P)#1/derivative(EP1,P,dx=1e-12)#1e-16

def PEderdm(P): # Speed of sound of Dark Matter
    return 1/derivativef5(P)#1/derivative(ex,P).df#f4(P)

def dy(r,y,F,Q): # This is for the tidal polarizability
    return(-y**2-y*F-Q)/r

def Pdmx(po,px,fx): # Find the pressure of DM in terms of total matter given the fraction and p_b
    fxsol = lambda px,po,fx : fx - ((px)/(po+px))
    ysolution = root(fxsol,px,args=(po,fx),method='lm')
    return ysolution.x[0]

def surf_star(t,y): # Optional equation to terminate the TOV
    return y[0]-4*10**(-10)

surf_star.terminal = True

#------------------ The TOV equations for all kinds of matter -----------------------

def TOV1(r,z):
    E = EP1(z[0])
    Fb = F(r,E,z[1],z[0])
    dPdE = PEderom1(z[0])
    Qb = Q(r,E,z[1],z[0],dPdE)
    return [dP(r,z[0],z[1],E),dM(r,E),dy(r,z[2],Fb,Qb)]

def TOV1x(r,z):
    E = ex(z[0])
    Fb = F(r,E,z[1],z[0])
    dPdE = PEderdm(z[0])
    Qb = Q(r,E,z[1],z[0],dPdE)
    return [dP(r,z[0],z[1],E),dM(r,E),dy(r,z[2],Fb,Qb)]

def TOV2f1(r,z):
    Eom = EP1(z[0])
    Edm = ex(z[2])
    M = z[1]+z[3]
    F = Fr(r,Eom,Edm,M,z[0],z[2])
    dPdE = PEderom1(z[0])
    dPdEdm = PEderdm(z[2])
    Q = rQ(r,Eom,M,z[0],Edm,z[2],dPdE,dPdEdm)
    return [dPom(r,z[0],z[2],M,Eom),dMom(r,Eom),dPdm(r,z[0],z[2],M,Edm),dMdm(r,Edm),dy(r,z[4],F,Q)]

# -------- Functions that find the number density and total number of particles for Dark Matter ----------

def n_x_finder(e_x_val): #find the n_x values (number density)
    n_x_r = lambda n_x_v : e_x_val - mx**4/(hc**3*8*np.pi**2) - x(n_x_v)*np.sqrt(1+x(n_x_v)**2)*(1+2*x(n_x_v)**2)-np.log(x(n_x_v)+np.sqrt(1+x(n_x_v)**2)) - (n_x_v**2*hc**3)/(2*z**2)
    n_x_value = fsolve(n_x_r, e_x_val - 10**(-12))
    return n_x_value

def N_x_Finder(R_x,r_x_data,n_data,r_tot_data,M_data): #find the N_x (total number of particles)
    epsilon = 1e-6
    x_n_modified = [r_x_data[0]]
    y_n_modified = [n_data[0]]
    z_m_modified = [M_data[0]]
    r_m_modified = [r_tot_data[0]]
    for i in range(1, len(r_x_data)):
        if r_x_data[i] <= x_n_modified[-1]:
            x_n_modified.append(x_n_modified[-1] + epsilon)
        else:
            x_n_modified.append(r_x_data[i])
        y_n_modified.append(n_data[i])
    for i in range(1, len(r_tot_data)):
        if r_tot_data[i] <= r_m_modified[-1]:
            r_m_modified.append(r_m_modified[-1] + epsilon)
        else:
            r_m_modified.append(r_tot_data[i])
        z_m_modified.append(M_data[i])
    m_f = PchipInterpolator(r_m_modified,z_m_modified)
    n_f = PchipInterpolator(x_n_modified,y_n_modified)
    I = lambda r: n_f(r)*(4*np.pi*r**2)/(np.sqrt(1-2.948*m_f(r)/r))
    Integral,_ = fixed_quad(I,0,R_x)
    return Integral

# Single Configuration TEST-DRIVE
# Run the TOV and get the results
ein = np.array([])
eout = np.array([])
h = 0.0001
y10=2
p_b = 150

p0 = p_b

#61.6#84.15#82.35
E0 = EP1(p0)

m1=(11.2*(10**(-6))/3)*(h**3)*EP1(p0)

p_x = 20

Pxc = p_x

#Pdmx(p0,0.5,fr1)
mdar = (11.2*(10**(-6))/3)*(h**3)*ex(Pxc)
y0 = [p0,m1,Pxc,mdar,y10]
sol = solve_ivp(TOV2f1,[0.1,10000],y0)#,events = surf_star)
rad1 = sol.t[len(sol.t)-1]
mass1 = sol.y[1,len(sol.y[1])-1] + sol.y[3,len(sol.y[3])-1]
yR = sol.y[4,len(sol.y[4])-1]

if sol.y[2,len(sol.y[2])-1]<sol.y[0,len(sol.y[0])-1]:
    print("in first")
    p0 = sol.y[0,len(sol.y[0])-1]
    m1 = sol.y[1,len(sol.y[1])-1] + sol.y[3,len(sol.y[3])-1]
    f = TOV1

elif sol.y[0,len(sol.y[0])-1]<sol.y[2,len(sol.y[2])-1]:
    print("in second")
    f = TOV1x
    p0 = sol.y[2,len(sol.y[0])-1]
    m1 = sol.y[1,len(sol.y[1])-1] + sol.y[3,len(sol.y[3])-1]

rmat = sol.t

y0 = [p0,m1,yR]
soln = solve_ivp(f,[rad1,100000],y0,events = surf_star)#,events=surf_star)

rad = soln.t[len(soln.t)-1]
mass2 = soln.y[1,len(soln.y[1])-1]
#Now I have to wrtie 2 if conditions. if and elif to append the right parameters in the right arrays.
if sol.y[2,len(sol.y[2])-1]<sol.y[0,len(sol.y[0])-1]:
    massn = mass2-sol.y[3,len(sol.y[3])-1]
    pin = sol.y[2]
    peksw = "Neutron"
    pmesa = "Dark"
    pout = np.append(sol.y[0],soln.y[0])


elif sol.y[0,len(sol.y[0])-1]<sol.y[2,len(sol.y[2])-1]:
    massx = sol.y[3,len(sol.y[3])-1] + (mass2-sol.y[1,len(sol.y[1])-1]) 
    pin = sol.y[0]
    pmesa = "Neutron"
    peksw = "Dark"
    pout = np.append(sol.y[2],soln.y[0])


rnmat = np.append(rmat,soln.t)

yR = soln.y[2,len(soln.y[2])-1]
Rs=2.948*mass2
beta=1.474*(mass2/rad)
kappa2=(8*beta**5/5)*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*(2*beta*(6-3*yR+3*beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta))**(-1)
lamda=64/3*kappa2*(rad/Rs)**5#(2/3)*kappa2*(rad/(1.474*mass))
    
# Put the results to separate arrays and find n_x and N_x
# Check which sol of TOV gives you the fluid you want

if sol.y[2,len(sol.y[2])-1]<sol.y[0,len(sol.y[0])-1]:
    massx = sol.y[3,len(sol.y[3])-1]
    massns = mass2-sol.y[3,len(sol.y[3])-1]
    p_2_mat = sol.y[2]
    p_1_mat = np.append(sol.y[0],soln.y[0])
    R_1 = soln.t[len(soln.t)-1]
    R_2 = sol.t[len(sol.t)-1]
    r_2_data = sol.t
    r_1_data = np.append(sol.t,soln.t)
    pin = sol.y[2]
    peksw = "Neutron Star"
    pmesa = "Dark Matter"
    pout = np.append(sol.y[0],soln.y[0])
    for i in range(0,len(pin)):
        ein = np.append(ein,ex(pin[i]))
    for i in range(0,len(pout)):
        eout = np.append(eout,EP1(pout[i]))


elif sol.y[0,len(sol.y[0])-1]<sol.y[2,len(sol.y[2])-1]:
    massx = mass2-sol.y[1,len(sol.y[1])-1]
    massns = sol.y[1,len(sol.y[1])-1]
    p_1_mat = sol.y[0]
    p_2_mat = np.append(sol.y[2],soln.y[0])
    R_2 = soln.t[len(soln.t)-1]
    R_1 = sol.t[len(sol.t)-1]
    r_1_data = sol.t
    r_2_data = np.append(sol.t,soln.t)
    pin = sol.y[0]
    pmesa = "Neutron Star"
    peksw = "Dark Matter"
    pout = np.append(sol.y[2],soln.y[0])
    for i in range(0,len(pin)):
        ein = np.append(ein,EP1(pin[i]))
    for i in range(0,len(pout)):
        eout = np.append(eout,ex(pout[i]))


# We also need e_x

e_x1 = EP1(p_1_mat)
e_x2 = ex(p_2_mat)

# Now we find n_x and store r and M data to arrays

n_x1 = f4(e_x1)
n_x2 = n_x_finder(e_x2)
r_tot_data = np.append(sol.t,soln.t)
M_data = np.append(sol.y[1]+sol.y[3],soln.y[1])
if sol.y[2,len(sol.y[2])-1]<sol.y[0,len(sol.y[0])-1]:
    r_tot_data1 = r_tot_data
    M_data1 = M_data
    r_tot_data2 = r_2_data
    M_data2 = sol.y[3]
elif sol.y[0,len(sol.y[0])-1]<sol.y[2,len(sol.y[2])-1]:
    r_tot_data2 = r_tot_data
    M_data2 = M_data
    r_tot_data1 = r_1_data
    M_data1 = sol.y[1]
# Now that we have the continuous n_x we proceed in finding N_x

N_x1 = N_x_Finder(R_1,r_1_data,n_x1,r_tot_data1,M_data1)
N_x2 = N_x_Finder(R_2,r_2_data,n_x2,r_tot_data2,M_data2)

# Now we are going to print the results.

print("The total number of fermionic1 matter particles is N$_{x1}$ = ",N_x1)
print("The total number of fermionic2 matter particles is N$_{x2}$ = ",N_x2)
print("M_tot , M_ns, M_x in the following by order:")
print(mass2)
print(massns)
print(massx)
print("pb and px:", p_b , p_x)
print("Radius:",rad)
print("Beta = ", beta)
print("Star radius:",rad1)
print("Lambda =",lamda)
print("k2 =",kappa2)
print("yR =",yR)

#--------------------- Plot to see the resulted P_b and P_x ----------------------------
plt.plot(rmat,pin,label = pmesa)
plt.plot(rnmat,pout,label = peksw)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

# Now we can use everything we got from above in a for loop to scan a grid of values of central pressure given a fraction of P_dm in terms of total P
fr1 = 10**(-2)#5*10**(-3)#5.49*10**(-1)#0.0889305#0.0011442#0.0303335#0.0222525#0.000995724#0.568842#0.0905297#0.00863409#0.0889305#0.0310265#0.0761623#0.00161952#0.00712679#0.46236559##0.173838#0.00259141#0.54959#0.00402442 #'''for mx100 z10'''0.1225318372696248#0.13089
#pmat1 = np.geomspace(0.001,40,num = 30,endpoint = False)
#pmat2 = np.geomspace(40,70,num = 30,endpoint = False)
#pmat3 = np.geomspace(70,1500,num = 30,endpoint = True)
#pmat = np.append(pmat1,pmat2)
#pmat = np.append(pmat,pmat3)
#pmat = append(pmat,70)
#pmat = np.sort(pmat)
pmat = np.geomspace(0.01,1500,num=150)
Rmatom1 = np.array([])
Mmatom1 = np.array([])
Lmat1 = np.array([])
Rmatin1 = np.array([])
Rmatinx1 = np.array([])
Mmatx1 = np.array([])
Mmatin1 = np.array([])
Mhalo1 = np.array([])
Rhalo1 = np.array([])
Rs1 = np.array([])
Ms1 = np.array([])
Md1 = np.array([])
p_dm = np.array([])
print("Start of loop 1")
for l in range (0,150,1):
    # Single Configuration TEST-DRIVE
    # Run the TOV and get the results
    ein = np.array([])
    eout = np.array([])
    h = 0.0001
    y10=2
    p_b = pmat[l]

    p0 = p_b

    #61.6#84.15#82.35
    E0 = EP1(p0)

    m1=(11.2*(10**(-6))/3)*(h**3)*EP1(p0)

    p_x = Pdmx(p0,0.5,fr1)
    Pxc = p_x
    print(p_b)
    print(p_x)
    #Pdmx(p0,0.5,fr1)
    mdar = (11.2*(10**(-6))/3)*(h**3)*ex(Pxc)
    y0 = [p0,m1,Pxc,mdar,y10]
    sol = solve_ivp(TOV2f1,[0.1,100000],y0,events = surf_star)
    rad1 = sol.t[len(sol.t)-1]
    mass1 = sol.y[1,len(sol.y[1])-1] + sol.y[3,len(sol.y[3])-1]
    yR = sol.y[4,len(sol.y[4])-1]

    if sol.y[2,len(sol.y[2])-1]<sol.y[0,len(sol.y[0])-1]:
        print("in first")
        p0 = sol.y[0,len(sol.y[0])-1]
        m1 = sol.y[1,len(sol.y[1])-1] + sol.y[3,len(sol.y[3])-1]
        f = TOV1

    elif sol.y[0,len(sol.y[0])-1]<sol.y[2,len(sol.y[2])-1]:
        print("in second")
        f = TOV1x
        p0 = sol.y[2,len(sol.y[0])-1]
        m1 = sol.y[1,len(sol.y[1])-1] + sol.y[3,len(sol.y[3])-1]

    rmat = sol.t

    y0 = [p0,m1,yR]
    soln = solve_ivp(f,[rad1,100000],y0,events = surf_star)#,events=surf_star)

    rad = soln.t[len(soln.t)-1]
    mass2 = soln.y[1,len(soln.y[1])-1]
    #Now I have to wrtie 2 if conditions. if and elif to append the right parameters in the right arrays.
    if sol.y[2,len(sol.y[2])-1]<sol.y[0,len(sol.y[0])-1]:
        massn = mass2-sol.y[3,len(sol.y[3])-1]
        pin = sol.y[2]
        peksw = "Neutron"
        pmesa = "Dark"
        pout = np.append(sol.y[0],soln.y[0])


    elif sol.y[0,len(sol.y[0])-1]<sol.y[2,len(sol.y[2])-1]:
        massx = sol.y[3,len(sol.y[3])-1] + (mass2-sol.y[1,len(sol.y[1])-1]) 
        pin = sol.y[0]
        pmesa = "Neutron"
        peksw = "Dark"
        pout = np.append(sol.y[2],soln.y[0])


    rnmat = np.append(rmat,soln.t)

    yR = soln.y[2,len(soln.y[2])-1]
    Rs=2.948*mass2
    beta=1.474*(mass2/rad)
    kappa2=(8*beta**5/5)*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*(2*beta*(6-3*yR+3*beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta))**(-1)
    lamda=64/3*kappa2*(rad/Rs)**5#(2/3)*kappa2*(rad/(1.474*mass))

    # Put the results to separate arrays and find n_x and N_x
    # Check which sol of TOV gives you the fluid you want

    if sol.y[2,len(sol.y[2])-1]<sol.y[0,len(sol.y[0])-1]:
        massx = sol.y[3,len(sol.y[3])-1]
        massns = mass2-sol.y[3,len(sol.y[3])-1]
        Ms1 = np.append(Ms1,massns)
        Md1 = np.append(Md1,massx)
        p_2_mat = sol.y[2]
        p_1_mat = np.append(sol.y[0],soln.y[0])
        R_1 = soln.t[len(soln.t)-1]
        R_2 = sol.t[len(sol.t)-1]
        r_2_data = sol.t
        r_1_data = np.append(sol.t,soln.t)
        pin = sol.y[2]
        peksw = "Neutron Star"
        pmesa = "Dark Matter"
        pout = np.append(sol.y[0],soln.y[0])
        for i in range(0,len(pin)):
            ein = np.append(ein,ex(pin[i]))
        for i in range(0,len(pout)):
            eout = np.append(eout,EP1(pout[i]))
        Rs1 = np.append(Rs1,rad)


    elif sol.y[0,len(sol.y[0])-1]<sol.y[2,len(sol.y[2])-1]:
        massx = mass2-sol.y[1,len(sol.y[1])-1]
        massns = sol.y[1,len(sol.y[1])-1]
        Ms1 = np.append(Ms1,massns)
        Md1 = np.append(Md1,massx)
        p_1_mat = sol.y[0]
        p_2_mat = np.append(sol.y[2],soln.y[0])
        R_2 = soln.t[len(soln.t)-1]
        R_1 = sol.t[len(sol.t)-1]
        r_1_data = sol.t
        r_2_data = np.append(sol.t,soln.t)
        pin = sol.y[0]
        pmesa = "Neutron Star"
        peksw = "Dark Matter"
        pout = np.append(sol.y[2],soln.y[0])
        for i in range(0,len(pin)):
            ein = np.append(ein,EP1(pin[i]))
        for i in range(0,len(pout)):
            eout = np.append(eout,ex(pout[i]))
        Rs1 = np.append(Rs1,rad1)


    # We also need e_x

    e_x1 = EP1(p_1_mat)
    e_x2 = ex(p_2_mat)

    # Now we find n_x and store r and M data to arrays
    print("FIND THE NX F4")
    #n_x1 = f4(e_x1)
    print("FIND THE NX NXFINDER")
    #n_x2 = n_x_finder(e_x2)
    print("DONE")
    r_tot_data = np.append(sol.t,soln.t)
    M_data = np.append(sol.y[1]+sol.y[3],soln.y[1])
    if sol.y[2,len(sol.y[2])-1]<sol.y[0,len(sol.y[0])-1]:
        r_tot_data1 = r_tot_data
        M_data1 = M_data
        r_tot_data2 = r_2_data
        M_data2 = sol.y[3]
    elif sol.y[0,len(sol.y[0])-1]<sol.y[2,len(sol.y[2])-1]:
        r_tot_data2 = r_tot_data
        M_data2 = M_data
        r_tot_data1 = r_1_data
        M_data1 = sol.y[1]
    # Now that we have the continuous n_x we proceed in finding N_x
    #print("FIND NX1FINDER")
    #N_x1 = N_x_Finder(R_1,r_1_data,n_x1,r_tot_data1,M_data1)
    #print("FIND NX1FINDER")
    #N_x2 = N_x_Finder(R_2,r_2_data,n_x2,r_tot_data2,M_data2)

    # Now we are going to print the results.

    #print("The total number of fermionic1 matter particles is N$_{x1}$ = ",N_x1)
    #print("The total number of fermionic2 matter particles is N$_{x2}$ = ",N_x2)
    print("M_tot , M_ns, M_x in the following by order:")
    print(mass2)
    print(massns)
    print(massx)
    print("pb and px:", p_b , p_x)
    print("Radius:",rad)
    Rmatom1 = np.append(Rmatom1,rad)
    Mmatom1 = np.append(Mmatom1,mass2)
    Lmat1 = np.append(Lmat1,lamda)
    p_dm = np.append(p_dm, Pxc)
    print(l)
pointmax1 = np.where(Mmatom1 == max(Mmatom1))

# Derive the Ordinary EoS M-R diagram
pmat = np.geomspace(0.001,1500,num = 200)
Rmatomns = np.array([])
Mmatomns = np.array([])
Lmatns = np.array([])
for i in range (0,200,1):
    h = 0.0001
    y10=2
    p0 = pmat[i]
    E0 = EP1(p0)
    m1=(11.2*(10**(-6))/3)*(h**3)*E0
    y0 = [p0,m1,y10]
    sol = solve_ivp(TOV1,[0.1,10000],y0,events = surf_star)

    mass2 = sol.y[1,len(sol.y[1])-1]
    rad = sol.t[len(sol.t)-1]

    yR = sol.y[2,len(sol.y[2])-1]
    Rs=2.948*mass2
    beta=1.474*(mass2/rad)
    kappa2=(8*beta**5/5)*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*(2*beta*(6-3*yR+3*beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta))**(-1)
    lamda=64/3*kappa2*(rad/Rs)**5#(2/3)*kappa2*(rad/(1.474*mass))
    
    Rmatomns = np.append(Rmatomns,rad)
    Mmatomns = np.append(Mmatomns,mass2)
    Lmatns = np.append(Lmatns,lamda)
    #print(rad," ", mass2)
    #print(np.sqrt(PEderom1(p0)))
    print("Iteration : ",i)
print("End of loop 1")

# Derive a Dark Star for the given mx and z

# DARK STAR
pmat = np.geomspace(0.0000001,1500,num = 50)
Rmatomds = np.array([])
Mmatomds = np.array([])
Lmatds = np.array([])
for i in range (0,50,1):
    h = 0.0001
    y10=2
    p0 = pmat[i]
    E0 = ex(p0)
    r0 = 0.0001
    m1= (11.2*(10**(-6))/3)*(h**3)*ex(p0)
    y0 = [p0,m1,y10]
    sol = solve_ivp(TOV1x,[0.1,100000],y0,events = surf_star)
    
    mass2 = sol.y[1,len(sol.y[1])-1]
    rad = sol.t[len(sol.t)-1]
    
    yR = sol.y[2,len(sol.y[2])-1]
    Rs=2.948*mass2
    beta=1.474*(mass2/rad)
    kappa2=(8*beta**5/5)*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*(2*beta*(6-3*yR+3*beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta))**(-1)
    lamda=64/3*kappa2*(rad/Rs)**5#(2/3)*kappa2*(rad/(1.474*mass))
    
    Rmatomds = np.append(Rmatomds,rad)
    Mmatomds = np.append(Mmatomds,mass2)
    Lmatds = np.append(Lmatds,lamda)
    print(i)

# Now we are going to plot the results to see the M-R diagram for the three different situations
matplotlib.use('TkAgg')
plt.rc('font', size=12)
plt.plot(Rmatomds,Mmatomds,'k--',label = "Dark Star")
plt.plot(Rmatom1,Mmatom1,'b',label = "Compact Object")#"NM+DM f = "+str(fr1))
plt.plot(Rmatomns,Mmatomns, 'g--', label = "Neutron Star")
#plt.plot(25.896,3.036,'rX') #This is in case you want to showcase a specific configuration
#The following is to add a few words for the configuration you choose the -star point
'''plt.text(0,4.2, "a)", fontsize=15,fontstyle='italic')
plt.text(0,2,"Compact Object\nParameters\n$M$ = 3.036 $M_{\odot}$\n $R$ = 25.896 $km$\n $m_x$ = 1500 MeV \n $y=0.033$ \n $f=5.49*10^{-1}$ \n $C=0.172$",fontsize = 12)
plt.text(20,2,r"${\bf X}$",color="red",fontsize=15)
plt.text(22,2,": $R_{star} = 8.64 km$",fontsize=15)
plt.text(22,1.8," $M_{star} = 0.63 M_{\odot}$",fontsize=15)
plt.xlabel("R$(km)$",fontsize = "18")
plt.ylabel("M $(M_\odot)$",fontsize = "18")'''
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(bbox_to_anchor=(0, 0.9),loc = 'upper left')
plt.grid()
#plt.savefig("myImage.png", format="png", dpi=1600)
plt.show()
