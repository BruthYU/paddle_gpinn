import paddlescience as psci
import numpy as np
geo=psci.geometry.Rectangular(origin=(-0.05, -0.05), extent=(0.05, 0.05))
geo.add_boundary(name="top", criteria=lambda x, y: abs(y - 0.05) < 1e-5)
geo.add_boundary(name="down", criteria=lambda x, y: abs(y + 0.05) < 1e-5)
geo.add_boundary(name="left", criteria=lambda x, y: abs(x + 0.05) < 1e-5)
geo.add_boundary(name="right", criteria=lambda x, y: abs(x - 0.05) < 1e-5)
npoints = 10201
geo_disc = geo.discretize(npoints=npoints, method="uniform")
pde = psci.pde.NavierStokes(
    nu=0.01, rho=1.0, dim=2, time_dependent=False, weight=0.0001)
weight_top_u = lambda x, y: 1.0 - 20.0 * abs(x)
bc_top_u = psci.bc.Dirichlet('u', rhs=1.0, weight=weight_top_u)
bc_top_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_down_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_down_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_left_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_right_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_right_v = psci.bc.Dirichlet('v', rhs=0.0)

pde.add_bc("top", bc_top_u, bc_top_v)
pde.add_bc("down", bc_down_u, bc_down_v)
pde.add_bc("left", bc_left_u, bc_left_v)
pde.add_bc("right", bc_right_u, bc_right_v)

pde_disc = pde.discretize(geo_disc=geo_disc)



net = psci.network.FCNet(
    num_ins=2,
    num_outs=3,
    num_layers=10,
    hidden_size=20,
    activation='tanh')

loss = psci.loss.L2(p=2)
algo = psci.algorithm.PINNs(net=net, loss=loss)
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solution = solver.solve(num_epoch=3000) # 30000
psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=solution)