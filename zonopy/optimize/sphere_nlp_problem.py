import torch
import numpy as np
from zonopy.kinematics.SO import make_spheres

class OfflineArmtdSphereConstraints:
    def __init__(self, dimension = 3, dtype = torch.float, device=None):
        self.dimension = dimension
        self.dtype = dtype
        self.np_dtype = torch.empty(0,dtype=dtype).numpy().dtype
        if device is None:
            device = torch.empty(0,dtype=dtype).device
        self.device = device
    
    def set_params(self, joint_occ, joint_pairs, g_ka, n_spheres_per_link, n_time, n_params):
        self.joint_occ = joint_occ
        self.joint_pairs = joint_pairs
        self.g_ka = g_ka

        n_joints = len(joint_occ)
        n_pairs = len(joint_pairs)
        self.n_spheres = n_spheres_per_link
        self.n_time = n_time
        self.n_params = n_params
        self.joint_occ = joint_occ
        self.joint_pairs = joint_pairs
        self.total_spheres = n_joints*n_time + n_spheres_per_link*n_pairs*n_time
        
        ### COMPAT
        self.M = self.total_spheres
        self.n_obs_in_FO = 1
        ###

        self.centers = torch.empty((self.total_spheres, self.dimension), dtype=self.dtype, device=self.device)
        self.radii = torch.empty((self.total_spheres), dtype=self.dtype, device=self.device)
        self.center_jac = torch.empty((self.total_spheres, self.dimension, self.n_params), dtype=self.dtype, device=self.device)
        self.radii_jac = torch.empty((self.total_spheres, self.n_params), dtype=self.dtype, device=self.device)

    def NN_fun(self, points):
        return torch.zeros(points.shape[0])+2, torch.zeros(points.shape).unsqueeze(-1)

    def __call__(self, x, Cons_out=None, Jac_out=None):
        x = torch.as_tensor(x, dtype=self.dtype)
        if Cons_out is None:
            Cons_out = np.empty(self.total_spheres, dtype=self.np_dtype)
        if Jac_out is None:
            Jac_out = np.empty((self.total_spheres, self.n_params), dtype=self.np_dtype)

        # First process with all the joints
        joints = {}
        eidx = 0
        for name, (pz, r) in self.joint_occ.items():
            sidx = eidx
            eidx = sidx + self.n_time
            self.centers[sidx:eidx] = pz.center_slice_all_dep(x)
            self.center_jac[sidx:eidx] = pz.grad_center_slice_all_dep(x)
            self.radii[sidx:eidx] = r
            self.radii_jac[sidx:eidx] = 0
            joints[name] = (self.centers[sidx:eidx], r, self.center_jac[sidx:eidx])

        # Now process all inbetweens
        for joint1, joint2 in self.joint_pairs:
            sidx = eidx
            eidx = sidx + self.n_spheres*self.n_time
            p1, r1, jac1 = joints[joint1]
            p2, r2, jac2 = joints[joint2]
            spheres = make_spheres(p1, p2, r1, r2, jac1, jac2, self.n_spheres)
            self.centers[sidx:eidx] = spheres[0].reshape(-1,self.dimension)
            self.radii[sidx:eidx] = spheres[1].reshape(-1)
            self.center_jac[sidx:eidx] = spheres[2].reshape(-1,self.dimension,self.n_params)
            self.radii_jac[sidx:eidx] = spheres[3].reshape(-1,self.n_params)
        
        # Do what you need with the centers and radii
        # NN(centers) - r > 0
        # D_NN(c(k)) -> D_c(NN) * D_k(c) - D_k(r)
        # D_c(NN) should have shape (n_spheres, 3, 1)
        # D_k(c) has shape (n_spheres, 3, n_params)
        # D_k(r) has shape (n_spheres, n_params)
        dist, dist_jac = self.NN_fun(self.centers)
        Cons_out[:] = -(dist - self.radii).cpu().numpy()
        Jac_out[:] = -(torch.einsum('sdi,sdp->sp', dist_jac, self.center_jac) - self.radii_jac).cpu().numpy()
        
        return Cons_out, Jac_out
