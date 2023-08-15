# TODO IMPLEMENT PRISMATIC, DOCUMENT, & CLEAN IMPORTS

import torch
from zonopy.transformations.rotation import get_pz_rotations_from_q
from zonopy.conSet.polynomial_zonotope.utils import remove_dependence_and_compress
import zonopy as zp

from zonopy.trajectories import BernsteinArmTrajectory
import numpy as np
from collections import OrderedDict

class JrsGenerator:
    def __init__(self,
                 robot,
                 traj_class=BernsteinArmTrajectory,
                 param_center=0,
                 param_range=np.pi/36,
                 tplan=0.5,
                 tfinal=1.0,
                 tdiscretization=0.01,
                 ultimate_bound=None,
                 k_r=None,
                 batched=False,
                 unique_tid=True,
                 verbose=False):
        
        self.verbose = verbose

        self.robot = robot
        self.traj = traj_class

        self.num_q = robot.num_q

        self.param_range = np.ones(self.num_q) * param_range
        self.param_range = np.vstack(
            [param_center - self.param_range, param_center + self.param_range]
        )

        self.tplan = tplan
        self.tfinal = tfinal
        self.tdiscretization = tdiscretization
        num_t = int(tfinal/tdiscretization)
        self.num_t = num_t

        self.id_map = OrderedDict()

        self.ultimate = (ultimate_bound, k_r) if ultimate_bound is not None and k_r is not None else None

        # Create base PZ's for parameters
        self.base_params = np.empty(self.num_q, dtype=object)
        self.k_ids = np.empty(self.num_q, dtype=int)
        for i in range(self.num_q):
            id = len(self.id_map)
            self.id_map[f'k{i}'] = id
            self.base_params[i] = zp.polyZonotope(
                [[0],[1]],
                1, id=id
            )
            self.k_ids[i] = id

        # Create base PZ's for time
        self.batched = batched
        if batched and unique_tid:
            import warnings
            warnings.warn("Using batched online JRS generation with unique_tid isn't recommended and may be slower. Consider setting unique_tid=False.")
        # batched = False
        # So time id's do not need to be unique
        if batched:
            i_s = np.arange(num_t)
            if unique_tid:
                ids = i_s + len(self.id_map)
                expMat = torch.eye(num_t, dtype=int)
                gens = torch.eye(num_t) * self.tdiscretization/2
                n_dep_gens = num_t
            else:
                ids = [len(self.id_map)]
                expMat = torch.eye(1, dtype=int)
                gens = torch.ones(num_t) * self.tdiscretization/2
                n_dep_gens = 1
            # Some tricks to make it into a batched poly zono
            centers = torch.as_tensor(
                self.tdiscretization*i_s+self.tdiscretization/2, 
                dtype=torch.get_default_dtype()
                )
            z = torch.vstack([centers, gens]).unsqueeze(2).transpose(0,1)
            self.times = zp.batchPolyZonotope(z, n_dep_gens, expMat, ids)
            # Make sure to update the id map (fixes slicing bug)!
            idmap = OrderedDict((f't{i}', id) for i,id in enumerate(ids))
            self.id_map.update(idmap)
        else:
            self.times = np.empty(num_t, dtype=object)
            if not unique_tid:
                id = len(self.id_map)
                self.id_map[f't0'] = id
            for i in range(num_t):
                if unique_tid:
                    id = len(self.id_map)
                    self.id_map[f't{i}'] = id
                self.times[i] = zp.polyZonotope(
                    [[self.tdiscretization*i+self.tdiscretization/2],[self.tdiscretization/2]],
                    1, id=id
                )
        
        # Create PZ's for error
        self.error_vel = None
        self.error_pos = None
        if self.ultimate is not None:
            self.error_pos = np.empty(self.num_q, dtype=object)
            self.error_vel = np.empty(self.num_q, dtype=object)
            for i in range(self.num_q):
                id = len(self.id_map)
                self.id_map[f'e_pos{i}'] = id
                self.error_pos[i] = zp.polyZonotope(
                    [[0],[ultimate_bound/k_r]], 1, id=id
                )

                id = len(self.id_map)
                self.id_map[f'e_vel{i}'] = id
                self.error_vel[i] = zp.polyZonotope(
                    [[0],[2*ultimate_bound]], 1, id=id
                )
    
    def gen_JRS(self, q_in, qd_in, qdd_in=None, taylor_degree=1, make_gens_independent=True, only_R=False):
        q_in = q_in.squeeze()
        qd_in = qd_in.squeeze()
        qdd_in = qdd_in.squeeze() if qdd_in is not None else None

        # create the trajectory generator
        gen = self.traj(
            q_in, qd_in, qdd_in,
            self.base_params, self.param_range,
            self.tplan, self.tfinal
        )

        # If only R, do a mini version
        if only_R:
            return self._gen_JRS_only_R(gen, taylor_degree, make_gens_independent)

        # Get the reference trajectory
        if self.verbose: print('Creating Reference Trajectory')
        q_ref, qd_ref, qdd_ref = gen.getReference(self.times)

        if self.batched:
            # Chose the right function
            rot_from_q = get_pz_rotations_from_q
        else:
            # Vectorize over q
            rot_from_q = np.vectorize(get_pz_rotations_from_q, excluded=[1,'taylor_deg'])
            q_ref, qd_ref, qdd_ref = q_ref.T, qd_ref.T, qdd_ref.T

        if self.verbose: print('Creating Reference Rotatotopes')
        R_ref = np.empty_like(q_ref)
        for i in range(self.num_q):
            R_ref[i] = rot_from_q(q_ref[i], self.robot.joint_axis[i], taylor_deg=taylor_degree)

        # Add tracking error if provided
        if self.ultimate is not None:
            if self.verbose: print('Adding tracking error to make tracked trajectory')
            q = (q_ref.T + self.error_pos).T
            qd = (qd_ref.T + self.error_vel).T
            qd_aux = (qd_ref.T + self.ultimate[1] * self.error_pos).T
            qdd_aux = (qdd_ref.T + self.ultimate[1] * self.error_vel).T
            if self.verbose: print('Creating Output Rotatotopes')
            R = np.empty_like(q)
            for i in range(self.num_q):
                R[i] = rot_from_q(q[i], self.robot.joint_axis[i], taylor_deg=taylor_degree)

        # Make independence if requested
        if make_gens_independent:
            if self.verbose: print("Making non-k generators independent")
            rem_dep = np.vectorize(remove_dependence_and_compress, excluded=[1])
            q_ref = rem_dep(q_ref, self.k_ids)
            qd_ref = rem_dep(qd_ref, self.k_ids)
            qdd_ref = rem_dep(qdd_ref, self.k_ids)
            R_ref = rem_dep(R_ref, self.k_ids)
            if self.ultimate is not None:
                q = rem_dep(q, self.k_ids)
                qd = rem_dep(qd, self.k_ids)
                qd_aux = rem_dep(qd_aux, self.k_ids)
                qdd_aux = rem_dep(qdd_aux, self.k_ids)
                R = rem_dep(R, self.k_ids)

        # Filler
        if self.ultimate is None:
            q, qd = q_ref, qd_ref
            qd_aux, qdd_aux = qd_ref, qdd_ref
            R = R_ref

        # Return as dict
        return {
            'q_ref': q_ref.T,
            'qd_ref': qd_ref.T,
            'qdd_ref': qdd_ref.T,
            'R_ref': R_ref.T,
            'q': q.T,
            'qd': qd.T,
            'qd_aux': qd_aux.T,
            'qdd_aux': qdd_aux.T,
            'R': R.T
        }
    
    def _gen_JRS_only_R(self, traj_gen, taylor_degree=1, make_gens_independent=True):
        if self.verbose: print('Only Generating R, Creating Reference Trajectory')
        q_ref = traj_gen.getReference(self.times)[0]

        if self.batched:
            # Chose the right function
            rot_from_q = get_pz_rotations_from_q
        else:
            # Vectorize over q
            rot_from_q = np.vectorize(get_pz_rotations_from_q, excluded=[1,'taylor_deg'])
            q_ref = q_ref.T
        
        # Add tracking error if provided
        if self.ultimate is not None:
            if self.verbose: print('Adding tracking error to make tracked trajectory')
            q = (q_ref.T + self.error_pos).T
        else:
            q = q_ref
        
        # Create the rotatotope
        if self.verbose: print('Creating Output Rotatotopes')
        R = np.empty_like(q)
        for i in range(self.num_q):
            R[i] = rot_from_q(q[i], self.robot.joint_axis[i], taylor_deg=taylor_degree)
        
        # Make independence if requested
        if make_gens_independent:
            if self.verbose: print("Making non-k generators independent")
            rem_dep = np.vectorize(remove_dependence_and_compress, excluded=[1])
            R = rem_dep(R, self.k_ids)
        return R.T


if __name__ == '__main__':
    q = torch.tensor([0])
    dq = torch.tensor([torch.pi])
    Q,R = gen_JRS(q,dq,joint_axes=None,taylor_degree=1,make_gens_independent=False)
    import pdb; pdb.set_trace()
    

    PZ_JRS,_ = zp.load_JRS_trig(q,dq)
    ax = zp.plot_polyzonos(PZ_JRS,plot_freq=1,edgecolor='blue',hold_on=True)
    #zp.plot_polyzonos(PZ_JRS,plot_freq=1,ax=ax)
    
    zp.plot_JRSs(Q,deg=1,plot_freq=1,ax=ax)
    
    
    