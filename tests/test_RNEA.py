import torch
import zonopy as zp
from zonopy.dynamics.RNEA import poly_zono_rnea
from zonopy.transformations.rotation import gen_rotatotope_from_jrs
import matplotlib.pyplot as plt
qpos =  torch.tensor([torch.pi/2,0,torch.pi/2,0,torch.pi/2,0])
qvel =  torch.tensor([1.2,1.2,1.2,1.2,1.2,1.2])

_,_,pz_params = zp.load_poly_zono_params('fetch',)
n_joints = pz_params['n_joints']
uniform_bound = 1.03
Kr = 1.01*torch.eye(n_joints)

q = torch.Tensor([0.0574,-0.0541,-1.4425,-1.8106,1.3694,-1.0811,-0.8970])
qd = torch.Tensor([-1.0795, -2.1729, -0.7759, 2.3958, 1.5403, -0.9605, 1.7959])
qd_a = torch.Tensor([-1.1237,-2.1938,-0.7666,2.3657,1.5597,-1.0046,1.7811])
qdd = torch.Tensor([0.2014,-1.1781,0.8960,-1.1521,-1.8074,-2.2207,-0.9181])
q_pz, qd_pz, qd_a_pz, qdd_pz = [],[],[],[]
R_pz, R_t_pz =[],[]
for i in range(n_joints):
    q_pz.append(zp.polyZonotope(q[i].reshape(1)))
    R_temp = gen_rotatotope_from_jrs(q_pz[-1],pz_params['joint_axes'][i])        
    R_pz.append(R_temp)
    R_t_pz.append(R_temp.T)
    qd_pz.append(zp.polyZonotope(qd[i].reshape(1)))
    qd_a_pz.append(zp.polyZonotope(qd_a[i].reshape(1)))
    qdd_pz.append(zp.polyZonotope(qdd[i].reshape(1)))    

f_pz, n_pz, u_pz = poly_zono_rnea(R_pz, R_t_pz, qd_pz, qd_a_pz, qdd_pz, pz_params)

'''
q, qd, qd_a, qdd_a, R, R_t = zp.load_traj_JRS_trig(qpos,qvel,uniform_bound,Kr)
n_time_steps = len(q)
n_time_steps = 1
f,n,u = [],[],[]

for t in range(n_time_steps):

    f_temp, n_temp, u_temp = poly_zono_rnea(R[t], R_t[t], qd[t], qd_a[t], qdd_a[t], pz_params)
    f.append(f_temp)
    n.append(n_temp)
    u.append(u_temp)
'''






import pdb; pdb.set_trace()
plt.figure()
ax = plt.gca()
f_pz[0].to_zonotope().plot(ax)
ax.autoscale(True)
plt.show()

#zp.plot_polyzonos(f)

#print(f[0])