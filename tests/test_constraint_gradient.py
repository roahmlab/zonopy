import torch 
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy
eps = 1e-6
n_links = 2
E = eps*torch.eye(n_links)

### 1. FO constraint gradient 
torch.set_default_dtype(torch.float64)

params = {'joint_axes':[torch.tensor([0.0,0.0,1.0])]*n_links,
        'R': [torch.eye(3)]*n_links,
        'P': [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(n_links-1),
        'H': [torch.eye(4)]+[torch.tensor([[1.0,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]*(n_links-1),
        'n_joints':n_links}

link_zonos = [zp.zonotope(torch.tensor([[0.5,0.5,0.0],[0.0,0.0,0.01],[0.0,0.0,0.0]]).T).to_polyZonotope()]*n_links

qpos = torch.pi*(2*torch.rand(n_links)-1)
qvel = torch.pi*(2*torch.rand(n_links)-1)

_,R_trig=zp.load_batch_JRS_trig(qpos,qvel)
FO_link,_, _ = forward_occupancy(R_trig,link_zonos,params)

FO_link = [fo_link.project([0,1]) for fo_link in FO_link]

lamb = (2*torch.rand(1,n_links)-1).repeat(100,1)

obs_Z = torch.tensor([[0.5,0.5],[1,0],[0,1]]).unsqueeze(0).repeat(100,1,1)


for j in range(n_links):
    A, b = zp.batchZonotope(torch.cat((obs_Z,FO_link[j].Grest),-2)).polytope()
    c_k = FO_link[j].center_slice_all_dep(lamb)
    cons, ind = torch.max((A@c_k.unsqueeze(-1)).squeeze(-1) - b,-1) # shape: n_timsteps, SAFE if >=1e-6
    grad_c_k = FO_link[j].grad_center_slice_all_dep(lamb)
    grad_cons = (A.gather(-2,ind.reshape(100,1,1).repeat(1,1,2))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6

    
    grad_c_k1 = torch.zeros(100,2,n_links)
    grad_c_k2 = torch.zeros(100,2,n_links)

    grad_cons1 = torch.zeros(100,n_links)
    grad_cons2 = torch.zeros(100,n_links)

    for i in range(n_links):
        c_k1 = FO_link[j].center_slice_all_dep(lamb+E[i])
        grad_c_k1[:,:,i] = (c_k1-c_k)/eps


        c_k2 = FO_link[j].center_slice_all_dep(lamb-E[i])
        grad_c_k2[:,:,i] = (c_k1-c_k2)/(2*eps)


    for i in range(n_links):
        grad_cons1 = (A.gather(-2,ind.reshape(100,1,1).repeat(1,1,2))@grad_c_k1).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
        grad_cons2 = (A.gather(-2,ind.reshape(100,1,1).repeat(1,1,2))@grad_c_k2).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6



    if ((grad_c_k-grad_c_k1)>eps).any():
        print(f'grad ck 1 for {j+1}-th FO doesnt match.')
    else:
        print(f'grad ck 1 for {j+1}-th FO is fine.')
    if ((grad_c_k-grad_c_k2)>eps).any():
        print(f'grad ck 2 for {j+1}-th FO doesnt match.')
    else:
        print(f'grad ck 2 for {j+1}-th FO is fine.')

    if ((grad_cons-grad_cons1)>eps).any():
        print(f'grad constraint 1 for {j+1}-th FO doesnt match.')
    else:
        print(f'grad constraint 1 for {j+1}-th FO is fine.')
    if ((grad_cons-grad_cons2)>eps).any():
        print(f'grad constraint 2 for {j+1}-th FO doesnt match.')
    else:
        print(f'grad constraint 2 for {j+1}-th FO is fine.')