##### 3D Rayleigh-Bernard Convection Validation #################
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import h5py

# Parameters
#Dring = 1 
#Lx, Ly, Lz = 8*Dring, 8*Dring, 10*Dring # size of confinement
Lx, Ly, Lz = 4, 4, 1
Nx, Ny, Nz = 64, 64, 72 # grid size
dealias = 3/2
dtype = np.float64
#sigma_0 = 0.04
#R_0 = 0.50
#gamma_0 = 0.04
Rayleigh = 2e6
Prandtl = 1
stop_sim_time = 15
timestepper = d3.RK443
max_timestep = 0.07

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly))
zbasis = d3.Chebyshev(coords['z'], size=Nz, bounds=(0, Lz))

# Fields
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis, zbasis)) # velocity field
#f = dist.VectorField(coords, name='f', bases=(xbasis, ybasis, zbasis)) # initial cartesian vorticity field
#f_polar = dist.Field(name='f_polar', bases=(xbasis,ybasis,zbasis)) # initial polar vorticity field
omega = dist.VectorField(coords, name='omega', bases=(xbasis, ybasis, zbasis)) # cartesian vorticity field (general)
ex, ey, ez = coords.unit_vector_fields(dist)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis,ybasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis,ybasis))
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis)) # pressure field
T = dist.Field(name='T', bases=(xbasis,ybasis,zbasis)) # temperature field
tau_p = dist.Field(name='tau_p')
tau_T1 = dist.Field(name='tau_T1', bases=(xbasis,ybasis))
tau_T2 = dist.Field(name='tau_T2', bases=(xbasis,ybasis))

# Forcing
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
#xcen = Lx*0.5
#ycen = Ly*0.5
#zcen = Lz*0.25
#ablob = gamma_0/(np.pi*(sigma_0**2))
#S = ( (z-zcen)**2 +  (((x-xcen)**2 + (y-ycen)**2)**(1/2) - R_0)**2)**(1/2)
#f_polar['g'] = ablob*np.exp(-(S/gamma_0)**2)*0 # polar vorticity

#if (x == 0).any() and (y < 0).any():
	#theta = (np.pi)/2
#elif (x == 0).any() and (y > 0).any():
	#theta = -(np.pi)/2
#elif (x == 0).any() and (y == 0).any():
	#theta = (np.pi)/2
#else:
	#theta = np.arctan(y/x)
	
#f['g'][0] = -np.sin(theta)*f_polar['g'] # Cartesian x component of vorticity
#f['g'][1] = np.cos(theta)*f_polar['g'] # Cartesian y component of vorticity
#f['g'][2] = 0 # Cartesian z component of vorticity
#fx = f['g'][0]
#fy = f['g'][1]
#fz = f['g'][2]
#f_mag = np.sqrt(fx**2 + fy**2 + fz**2)
# Substitutions0
dz = lambda A: d3.Differentiate(A, coords['z'])
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
#ex = dist.VectorField(coords, name='ex')
#ey = dist.VectorField(coords, name='ey')
#ez = dist.VectorField(coords, name='ez')
#ex['g'][0] = 1
#ey['g'][1] = 1
#ez['g'][2] = 1
#u['g'] = 0
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_T = d3.grad(T) + ez*lift(tau_T1)# First-order reduction
dot = lambda A, B: d3.DotProduct(A, B)

# Poisson 
#Poisson = d3.LBVP([u, tau_u1], namespace=locals())
#Poisson.add_equation("lap(u) + tau_u1 = -curl(f)")
#Poisson.add_equation("u(z=0) = 0")

# Solver
#solver = Poisson.build_solver()
#solver.solve()

######################################################################################################################################################

# RBC 3D

# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, u, T, tau_u1, tau_u2, tau_T1, tau_T2, tau_p], namespace=locals()) #  
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - kappa*div(grad_T) + lift(tau_T2) = - u@grad(T)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - T*ez + lift(tau_u2) = - u@grad(u)") # 
problem.add_equation("T(z=0) = 1")
problem.add_equation("u(z=0) = 0")
problem.add_equation("T(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions	
T.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
T['g'] *= z * (Lz - z) # Damp noise at walls
T['g'] += Lz - z # Add linear background

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Gather global data
x = xbasis.global_grid()
y = ybasis.global_grid()
z = zbasis.global_grid()
#ug = u.allgather_data('g')
#fg = f.allgather_data('g')
X = np.zeros([Nx*Ny*Nz])
Y = np.zeros([Nx*Ny*Nz])
Z = np.zeros([Nx*Ny*Nz])

# Main loop
startup_iter = 10
for i in range(Nx):
	X[i] = x[i,0,0]
#    print(X[i])

for j in range(Ny):
	Y[j] = y[0,j,0]
#    print(Y[j])

for k in range(Nz):
	Z[k] = z[0,0,k]
	
omega = d3.curl(u)
        	    
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 5 == 0:
            count = solver.iteration
        if solver.sim_time != stop_sim_time:
            file1 = open("RBC/3D_RBC"f"{count}"".tec", "w")
            ux = u['g'][0]
            uy = u['g'][1]
            uz = u['g'][2]
            wx = omega.evaluate()['g'][0]
            wy = omega.evaluate()['g'][1]
            wz = omega.evaluate()['g'][2]
            W_mag = np.sqrt(wx**2 + wy**2 + wz**2)
            file1.write(' Variables="X","Y","Z","u","v","w","wx","wy","wz","W_mag","T"\n') #  
            file1.write(' Zone  I=64, J=64, K=72, F=POINT\n')
            for k in range(Nz):
            	for j in range(Ny):
                    for i in range(Nx):
                        file1.write(f" {X[i]}\t{Y[j]}\t{Z[k]}\t{ux[i,j,k]}\t{uy[i,j,k]}\t{uz[i,j,k]}\t{wx[i,j,k]}\t{wy[i,j,k]}\t{wz[i,j,k]}\t{W_mag[i,j,k]}\t{T['g'][i,j,k]}\n") # 
            file1.close()
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
