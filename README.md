# Sample Codes
## A note towards readers

You will find here the codes I have used throughout the years as a bachelors, masters and now a PhD student. You will find many similarities and argue that this is the same code again and again and you will not be wrong. The code is pretty much the same, but the parameters change and physics is derived through the parameters. We need to find the physical meaning behind the parameters and the results they give us and use it to explain our observational data not only from a mathematical, but also from a physics point of view. This is why the codes I use are only half of the actual research I am doing with the team in my PhD. For example, a simple minus sign instead of plus in the Energy Density of DM in front of the interaction term, drastically changes both the results and the physical meaning of the EoS and nature of DM one uses.

## Code files.py explanation
### TwoFluid_relativistic.py
Two_Fluid_relativistic.py is a code wirtten in Python that solves the TOV equations for a Two-Fluid model with a relativistic EoS for the DM. The way that works is by inserting a specific dm particle mass (m_x) and a specific interaction of DM particles (z) and two specific central pressures for DM and OM (Ordinary Matter). Also, specifying a fraction (f) for the DM pressure in terms of total pressure (when you want to derive the M-R diagram) helps you to have a relation between p_b and p_x (baryonic and dm pressure) in order to derive the M-R diagram because you need the two central pressures for each fluid.

Not only that, but I have also included the derivation of the M-R diagram for the NS-DM admixed configurations. It also includes the solution of TOV for an ordinary NS and a Dark Star and the plotting of the resulted M-R diagrams.

### TOV_Solver.py
The TOV_Solver.py code is a simple code that contains a class which solves the TOV equations for a single configuration and for deriving the M-R diagram. In more details: when creating an object you need to insert a filename (array-like EoS) as an argument. Now you can use two different methods for creating a single configuration, given the central pressure as argument, and another for deriving the M-R diagram, given the starting p_c, ending p_c and the number of points for the grid.


### EoS_maker.py
The EoS_maker.py contains the classes which you can use to create the EoS for neutron and dark matter. If you just want to create those EoS and use them afterwards, instead of searching the Two_Fluid.py or event the TOV_Solver.py for the simple EoS interpolation, I have gathered the necessary code and created three classes for the EoS creation. The Ordinary matter EoS is not exactly created, rather interpolated because you need a function in order to use at the TOV afterwards. So given the filename of the array-like data (shaped like rho|e_rho|p_b) you can receive the interpolated EoS in the end. For the second and third class, I added the DM EoS creation, relativistic and non-relativistic (in case the particle is too heavy and thus non-relativistic). Simply give the interaction z and the dm particle mass mx and then receive at the end the EoS function for dm.

For those who don't know what EoS is, it is simply a relation between the Energy density E and Pressure P, in simple terms the E(P) function which is needed for the TOV afterwards. (This can also work if you go P(E), but I choose to work with E(P)).

#### Note
Now of course you can use the EoS_maker.py classes in the TOV_Solver.py, the baryonic EoS in particular, use the NS_EoS as primary class and create other child classes that simply use the constructor of NS_EoS and have methods of data manipulation and TOV solutions. But I prefered to have the whole code there, in case someone misses the other files and just checks the TOV_Solver.py one. 
