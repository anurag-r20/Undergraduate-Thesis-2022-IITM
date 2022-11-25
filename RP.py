import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import h5py

# Parameters
Dring = 1 
Lx, Ly = 4*Dring, 10*Dring # size of confinement
Nx, Ny = 128, 256 # grid size
dealias = 3/2
dtype = np.float64
sigma_0 = 0.04
R_0 = 0
gamma_0 = 0.04
Rayleigh = 2e6
Prandtl = 1
stop_sim_time = 15
timestepper = d3.RK222
max_timestep = 0.0125

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx)) 
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly))

# Fields
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis)) # velocity field
omega = dist.VectorField(coords, name='omega', bases=(xbasis, ybasis)) # vorticity field
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
p = dist.Field(name='p', bases=(xbasis,ybasis)) # pressure field
T = dist.Field(name='T', bases=(xbasis,ybasis)) # temperature field
tau_p = dist.Field(name='tau_p')
tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
tau_T2 = dist.Field(name='tau_T2', bases=xbasis)

# Forcing
x, y = dist.local_grids(xbasis, ybasis)
f = dist.VectorField(coords, name='f', bases=(xbasis, ybasis)) # initial vorticity field
h = dist.VectorField(coords, name='h', bases=xbasis)
xcen = Lx*0.5
ycen = Ly*0.5
ablob = gamma_0/(np.pi*(sigma_0**2))
S = ((((x-xcen)**2 + (y-ycen)**2)**(1/2) - R_0)**2)**(1/2)
#S = ((y-ycen)**2 +((x-xcen) - R_0)**2)**(1/2)
f['g'] = ablob*np.exp(-(S/gamma_0)**2)
h['g'] = 0


# Substitutions
dy = lambda A: d3.Differentiate(A, coords['y'])
#lift_basis = ybasis.derivative_basis(1)
#lift = lambda A: d3.Lift(A, lift_basis, -1)
ex, ey = coords.unit_vector_fields(dist) # unit vetor definition
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
grad_u = d3.grad(u) # First-order reduction
grad_T = d3.grad(T) # First-order reduction

# Poisson 
Poisson = d3.LBVP([u, tau_u1], namespace=locals())
Poisson.add_equation("lap(u) + tau_u1 = -skew(f)")
Poisson.add_equation("u(y=0) = h")

# Solver
solver = Poisson.build_solver()
solver.solve()

######################################################################################################################################################

# RBC 2D 

# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, u, T, tau_p], namespace=locals()) #
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - kappa*div(grad_T) = - u@grad(T)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - T*ey  = - u@grad(u)") #
#problem.add_equation("T(y=0) = Ly")
#problem.add_equation("u(y=0) = 0")
#problem.add_equation("T(y=Ly) = 0")
#problem.add_equation("u(y=Ly) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
omegaz = f['g'][1]
X = np.zeros([Nx*Ny])
Y = np.zeros([Nx*Ny])
for i in range(Nx):
	for j in range(Ny):
		if(omegaz[i,j]>1e-1):
			T['g'][i,j]=1
		else:
			T['g'][i,j]=0

# Analysis though .HDF5 files

#snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
#snapshots.add_task(T, name='Temperature')
#snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
#snapshots.add_task(ex@u, name='u')


# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Gather global data
x = xbasis.global_grid()
y = ybasis.global_grid()
#ug = u.allgather_data('g')
#fg = f.allgather_data('g')
X = np.zeros([Nx*Ny])
Y = np.zeros([Nx*Ny])

# Main loop
startup_iter = 10
for i in range(Nx):
	X[i] = x[i,0]
#    print(X[i])

for j in range(Ny):
	Y[j] = y[0,j]
	
omega = d3.skew(u) # Instantaneous vorticity

try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 20 == 0:
            count = solver.iteration
        if solver.sim_time != stop_sim_time:
            file1 = open("2D_T"f"{count}"".tec", "w")
            ux = u['g'][0]
            uy = u['g'][1]
            wx = omega.evaluate()['g'][0]
            wy = omega.evaluate()['g'][1]
            file1.write(' Variables="X","Y","u","v","wx","wy","T"\n') # 
            file1.write(' Zone  I=128, J=256, F=POINT\n')
            for j in range(Ny):
            	for i in range(Nx):
                	file1.write(f"{X[i]}\t{Y[j]}\t{ux[i,j]}\t{uy[i,j]}\t{wx[i,j]}\t{wy[i,j]}\t{T['g'][i,j]}\n") # 
            file1.close()
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

