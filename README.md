# Sample-Codes

Two_Fluid_relativistic.py is a code wirtten in Python that solves the TOV equations for a Two-Fluid model with a relativistic EoS for the DM. The way that works is by inserting a specific dm particle mass (m_x) and a specific interaction of DM particles (z) and two specific central pressures for DM and OM (Ordinary Matter). Also, specifying a fraction (f) for the DM pressure in terms of total pressure (when you want to derive the M-R diagram) helps you to have a relation between p_b and p_x (baryonic and dm pressure) in order to derive the M-R diagram because you need the two central pressures for each fluid.

Not only that, but I have also included the derivation of the M-R diagram for the NS-DM admixed configurations. It also includes the solution of TOV for an ordinary NS and a Dark Star and the plotting of the resulted M-R diagrams.

The TOV_Solver.py code is a simple code that contains a class which solves the TOV equations for a single configuration and for deriving the M-R diagram. In more details: when creating an object you need to insert a filename (array-like EoS) as an argument. Now you can use two different methods for creating a single configuration, given the central pressure as argument, and another for deriving the M-R diagram, given the starting p_c, ending p_c and the number of points for the grid.
