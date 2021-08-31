import numpy as np
import random
import gsd.hoomd
import hoomd

cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu,seed = 1)
integrator = hoomd.hpmc.integrate.ConvexPolygon()


A =  [(-0.5223290993692602, -0.16516412776125292), (0.47767090063073975, -0.16516412776125292), (0.04465819873852046, 0.3303282555225058)]
B =  [(-0.5534779580863476, -0.2274853687848006), (0.44652204191365247, -0.2274853687848006), (0.33762976744895196, 0.32497372029196253), (-0.3079855748025888, 0.197719911933842)]
C =  [(-0.46196653872316695, -0.2417397096265952), (0.538033461276833, -0.2417397096265952), (0.28352584456059177, 0.1990804134392008), (-0.24936389836384554, 0.38100648201528425)]
D =  [(-0.7425143969508786, 0.00812615065931594), (0.12351100683355996, -0.49187384934068407), (0.6190033901173186, -0.05886114744846476), (0.12351100683355996, 0.508126150659316)]

integrator.shape['A'] = dict(vertices = A)
integrator.shape['B'] = dict(vertices = B)
integrator.shape['C'] = dict(vertices = C)
integrator.shape['D'] = dict(vertices = D)

simulation.operations.integrator = integrator
shape_areas = list(map(hoomd.dem.utils.area,[A,B,C,D]))
combined_area = np.sum(shape_areas)

Lx = 100
Ly = 100
box_area = Lx * Ly

particle_number = 500

particle_positions_ = []
particle_positions = []
for x in np.arange(-Lx/2,Ly/2,3.5):
    for y in np.arange(-Ly/2,Ly/2,3.5):
        particle_positions_.append([x,y,0])

for _ in range(particle_number):
    i = random.randint(0,len(particle_positions_)-1)
    p = particle_positions_.pop(i)
    particle_positions.append(p)


snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = particle_number
snapshot.particles.position = particle_positions
snapshot.particles.orientation = [(1,0,0,0)] * particle_number
snapshot.particles.typeid = [0,1,2,3] * (particle_number//4)
snapshot.particles.types = ['A','B','C','D']
snapshot.configuration.box = [Lx, Ly, 0, 0, 0, 0]

with gsd.hoomd.open(name='initial_lattice.gsd', mode='xb') as f:
    f.append(snapshot)

simulation.create_state_from_gsd(filename='initial_lattice.gsd')
simulation.run(10)
hoomd.write.GSD.write(state=simulation.state, mode='xb', filename='initial_randomized.gsd')

main_logger = hoomd.logging.Logger()
main_logger.add(integrator, quantities=['type_shapes'])
main_writer = hoomd.write.GSD(filename='trajectory.gsd',
                             trigger=hoomd.trigger.Periodic(1),
                             mode='xb',
                             filter=hoomd.filter.All(),
                             log=main_logger)
simulation.operations.writers.append(main_writer)

sdf = hoomd.hpmc.compute.SDF(dx =1e-4,xmax= 0.02)
sdf_logger = logger = hoomd.logging.Logger()
sdf_logger.add(sdf)
sdf_writer = hoomd.write.GSD(filename='log.gsd',
                             trigger=hoomd.trigger.Periodic(10),
                             mode='xb',
                             filter=hoomd.filter.Null(),
                             log = sdf_logger)
simulation.operations.writers.append(sdf_writer)
simulation.run(100)


# imA perp ker At.
# imAt perp ker A
# symmetric, self adjoint
# (etc..)
# boundary conditions, fourier conditions - same basis