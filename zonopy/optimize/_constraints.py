import torch

def collision_constraint(FO_link,obs,ka):
    n_joints = len(FO_link)
    n_obs = len(obs)
    
    for j in range(n_joints):
        for o in range(n_obs):
            buff = FO_link[j] - obs[o]
            c, c_grad = buff.slice_all_dep(ka)
            buff_z = buff.to_zonotope()
            A,b = buff_z.polytope()

            con_temp,ind =  -torch.max(A@c-b,axis=-2)
            con_grad_temp = A[ind]@c_grad - b
    return con_temp 



if __name__ == '__main__':
    import zonopy as zp
    N_joints = 7
    qpos =  torch.tensor([0.0]*N_joints)
    qvel =  torch.tensor([torch.pi/2]*N_joints)
    params = {'joint_axes':[torch.tensor([0.0,0.0,1.0])]*N_joints, 
            'R': [torch.eye(3)]*N_joints,
            'P': [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0])]*(N_joints-1),
            'H': [torch.eye(4)]+[torch.tensor([[1.0,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]*(N_joints-1),
            'n_joints':N_joints}
    link_zonos = [zp.zonotope(torch.tensor([[0.5,0.5,0.0],[0.0,0.0,0.01],[0.0,0.0,0.0]]).T).to_polyZonotope()]*N_joints # [1,0,0]

    _, R_trig = zp.load_JRS_trig(qpos,qvel)
    FO_link, r_trig, p_trig = [], [], []
    for t in range(100):
        FO_link_temp,r_temp,p_temp = zp.kinematics.FO.forward_occupancy(R_trig[t],link_zonos,params)
        FO_link.append(FO_link_temp)
        r_trig.append(r_temp)
        p_trig.append(p_temp)

    obs = [zp.zonotope([[1.5, 0.2, 0],[0.5, 0, 0.2]])]

    



