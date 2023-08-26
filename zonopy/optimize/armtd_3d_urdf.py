# TODO VALIDATE

import torch
import numpy as np
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy as forward_occupancy
import cyipopt


import time

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1.0
BUFFER_AMOUNT = 0.03

from urchin import URDF
from typing import List

class ARMTD_3D_planner():
    def __init__(self,
                 robot: URDF,
                 zono_order: int = 2, # this appears to have been 40 before but it was ignored for 2
                 max_combs: int = 200,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
                 ):
        self.dtype, self.device = dtype, device

        self.robot = robot
        self.PI = torch.tensor(torch.pi,dtype=self.dtype,device=self.device)
        self.JRS_tensor = zp.preload_batch_JRS_trig(dtype=self.dtype,device=self.device)
        
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.combs = self._generate_combinations_upto(max_combs)

        self._setup_robot(robot)

        # self.wrap_env(env)
        # self.n_timesteps = 100
        #self.joint_speed_limit = torch.vstack((torch.pi*torch.ones(n_links),-torch.pi*torch.ones(n_links)))
    
    def _setup_robot(self, robot):
        ## Duplicated from env
        self.dof = len(robot.actuated_joints)
        continuous_joints = []
        pos_lim = [[-np.Inf, np.Inf]]*self.dof
        vel_lim = [np.Inf]*self.dof
        eff_lim = [np.Inf]*self.dof
        joint_axis = []
        for i,joint in enumerate(robot.actuated_joints):
            if joint.joint_type == 'continuous':
                continuous_joints.append(i)
            elif joint.joint_type in ['floating', 'planar']:
                raise NotImplementedError
            if joint.limit is not None:
                lower = joint.limit.lower if joint.limit.lower is not None else -np.Inf
                upper = joint.limit.upper if joint.limit.upper is not None else np.Inf
                pos_lim[i] = [lower, upper]
                vel_lim[i] = joint.limit.velocity
                eff_lim[i] = joint.limit.effort
            joint_axis.append(joint.axis)
        self.joint_axis = torch.tensor(joint_axis)
        self.pos_lim = np.array(pos_lim).T
        self.vel_lim = np.array(vel_lim)
        # self.eff_lim = np.array(eff_lim) # Unused for now
        self.continuous_joints = np.array(continuous_joints, dtype=int)
        self.pos_lim_mask = np.isfinite(self.pos_lim).any(axis=0)
        pass

    def _generate_combinations_upto(self, max_combs):
        return [torch.combinations(torch.arange(i,device=self.device),2) for i in range(max_combs+1)]
    
    def _wrap_cont_joints(self, pos: torch.tensor) -> torch.tensor:
        pos = torch.clone(pos)
        pos[..., self.continuous_joints] = (pos[..., self.continuous_joints] + torch.pi) % (2 * torch.pi) - torch.pi
        return pos

    # def wrap_env(self,env):
    #     assert env.dimension == 3
    #     self.dimension = 3
    #     self.n_links = env.n_links
    #     self.n_obs = env.n_obs

    #     P,R,self.link_zonos = [], [], []
    #     for p,r,l in zip(env.P0,env.R0,env.link_zonos):
    #         P.append(p.to(dtype=self.dtype,device=self.device))
    #         R.append(r.to(dtype=self.dtype,device=self.device))
    #         self.link_zonos.append(l.to(dtype=self.dtype,device=self.device))
    #     self.params = {'n_joints':env.n_links, 'P':P, 'R':R}        
    #     self.joint_axes = env.joint_axes.to(dtype=self.dtype,device=self.device)
    #     self.vel_lim =  env.vel_lim.cpu()
    #     self.pos_lim = env.pos_lim.cpu()
    #     self.actual_pos_lim = env.pos_lim[env.lim_flag].cpu()
    #     self.n_pos_lim = int(env.lim_flag.sum().cpu())
    #     self.lim_flag = env.lim_flag.cpu()

    # def wrap_cont_joint_to_pi(self,phases):
    #     phases_new = torch.clone(phases)
    #     phases_new[~self.lim_flag] = (phases[~self.lim_flag] + torch.pi) % (2 * torch.pi) - torch.pi
    #     return phases_new

    # NOTE Assumes the base link can be ignored
    # THIS IS USELESS FOR ARMTD :(
    def _prepare_JLS_constraints(self,
                                 JRS_Q: List[zp.batchPolyZonotope],
                                 JRS_Qd: List[zp.batchPolyZonotope],
                                 ):
        JLS_gen_time = time.perf_counter()
        n_joints = len(JRS_Q)
        bounds_constraints = []
        for idx in range(n_joints):
            q, qd = JRS_Q[idx], JRS_Qd[idx]
            q_buff = torch.sum(torch.abs(q.Grest), dim=-2)
            q_ub = zp.batchPolyZonotope(torch.cat((q.c + q_buff, q.G), dim=-2), q.n_dep_gens, q.expMat, q.id)
            q_lb = zp.batchPolyZonotope(torch.cat((q.c - q_buff, q.G), dim=-2), q.n_dep_gens, q.expMat, q.id)
            active_q_ub = q_ub.to_interval().sup >= 0
            active_q_lb = q_lb.to_interval().sup >= 0
            qd_buff = torch.sum(torch.abs(qd.Grest), dim=-2)
            qd_ub = zp.batchPolyZonotope(torch.cat((qd.c + qd_buff, qd.G), dim=-2), qd.n_dep_gens, qd.expMat, qd.id)
            qd_lb = zp.batchPolyZonotope(torch.cat((qd.c - qd_buff, qd.G), dim=-2), qd.n_dep_gens, qd.expMat, qd.id)
            active_qd_ub = qd_ub.to_interval().sup >= 0
            active_qd_lb = qd_lb.to_interval().sup >= 0
            # Only add if active
            if active_q_ub.any():
                bounds_constraints.append(q_ub[active_q_ub])
            if active_q_lb.any():
                bounds_constraints.append(q_lb[active_q_lb])
            if active_qd_ub.any():
                bounds_constraints.append(qd_ub[active_qd_ub])
            if active_qd_lb.any():
                bounds_constraints.append(qd_lb[active_qd_lb])
        final_time = time.perf_counter()
        return np.array(bounds_constraints), final_time - JLS_gen_time
        
    def _prepare_FO_constraints(self,
                                JRS_R: zp.batchMatPolyZonotope,
                                obs_zono: zp.batchZonotope,
                                ):
        # constant
        n_obs = len(obs_zono)

        ### prepare the JRS
        # process_time = time.perf_counter()
        # _, JRS_R = zp.process_batch_JRS_trig(self.JRS_tensor,
        #                                      qpos,
        #                                      qvel,
        #                                      self.joint_axis)
        
        ### get the forward occupancy
        FO_gen_time = time.perf_counter()
        FO_links, _ = forward_occupancy(JRS_R, self.robot, self.zono_order)
        # let's assume we can ignore the base link and convert to a list of pz's
        # end effector link is a thing???? (n actuated_joints = 7, n links = 8)
        FO_links = list(FO_links.values())[1:]
        # two more constants
        n_links = len(FO_links)
        n_frs_timesteps = len(FO_links[0])

        ### begin generating constraints
        constraint_gen_time = time.perf_counter()
        out_A = np.empty((n_links),dtype=object)
        out_b = np.empty((n_links),dtype=object)
        out_g_ka = np.ones((self.dof),dtype=np.float32) * np.pi/24                  # Hardcoded because it's preloaded...
        out_FO_links = np.empty((n_links),dtype=object)

        # combine all obstacles
        # Batch dims = {0: n_obs, 1: n_timesteps}
        # obs_Z = []
        # for obs in obs_zono:
        #     obs_Z.append(obs.Z.unsqueeze(0))
        # obs_Z = torch.cat(obs_Z,0).to(dtype=self.dtype, device=self.device).unsqueeze(1).repeat(1,self.n_timesteps,1,1)
        obs_Z = obs_zono.Z.unsqueeze(1) # expand dim for timesteps

        obs_in_reach_idx = torch.zeros(n_obs,dtype=bool,device=self.device)
        for idx, FO_link_zono in enumerate(FO_links):
            # buffer the obstacle by Grest
            obs_buff_Grest_Z = torch.cat((obs_Z.expand(-1, n_frs_timesteps, -1, -1), FO_link_zono.Grest.expand(n_obs, -1, -1, -1)), -2)
            obs_buff_Grest = zp.batchZonotope(obs_buff_Grest_Z)
            # polytope with prefegenerated combinations
            # TODO: merge into the function w/ a cache
            A_Grest, b_Grest  = obs_buff_Grest.polytope(self.combs)

            # Further overapproximate to identify which obstacles we need to care about
            FO_overapprox_center = zp.batchZonotope(FO_link_zono.Z[...,:FO_link_zono.n_dep_gens+1, :].expand(n_obs, -1, -1, -1))
            obs_buff_overapprox = obs_buff_Grest + (-FO_overapprox_center)
            # polytope with prefegenerated combinations
            # TODO: merge into the function w/ a cache
            _, b_obs_overapprox = obs_buff_overapprox.reduce(3).polytope(self.combs)

            obs_in_reach_idx += (torch.min(b_obs_overapprox.nan_to_num(torch.inf),-1)[0] > -1e-6).any(-1)
            # only save the FO center
            # TODO: add dtype and device args
            out_FO_links[idx] = zp.batchPolyZonotope(FO_link_zono.Z[...,:FO_link_zono.n_dep_gens+1, :], FO_link_zono.n_dep_gens, FO_link_zono.expMat, FO_link_zono.id).cpu()
            out_A[idx] = A_Grest.cpu()
            out_b[idx] = b_Grest.cpu()

        # Select for the obs in reach only
        for idx in range(n_links):
            out_A[idx] = out_A[idx][obs_in_reach_idx].cpu()
            out_b[idx] = out_b[idx][obs_in_reach_idx].cpu()
        
        out_n_obs_in_frs = int(obs_in_reach_idx.sum())
        final_time = time.perf_counter()
        out_times = {
            'FO_gen': constraint_gen_time - FO_gen_time,
            'constraint_gen': final_time - constraint_gen_time,
        }
        return out_FO_links, out_A, out_b, out_g_ka, out_n_obs_in_frs, out_times

    def trajopt(self, qpos, qvel, qgoal, ka_0, FO_links, A, b, g_ka, n_obs_in_FO):
        n_links = len(FO_links)
        n_timesteps = len(FO_links[0])
        n_obs_cons = n_timesteps * n_obs_in_FO

        M_fo = n_links * n_obs_cons
        M_limits = 2*self.dof + 6*self.pos_lim_mask.sum()
        M = int(M_fo + M_limits)

        # Moved to another file
        from zonopy.optimize.nlp_setup_extracted_expanded import armtd_nlp
        problem_obj = armtd_nlp(
            qpos,
            qvel,
            qgoal,
            FO_links,
            A,
            b,
            g_ka,
            np.float32, # FIX
            n_links,
            self.dof,
            n_timesteps,
            n_obs_in_FO,
            M,
            M_fo,
            M_limits,
            self.pos_lim,
            self.vel_lim,
            self.continuous_joints,
            self.pos_lim_mask,
            3,
            n_obs_cons,
            T_PLAN,
            T_FULL,
        )

        nlp = cyipopt.Problem(
        n = self.dof,
        m = M,
        problem_obj=problem_obj,
        lb = [-1]*self.dof,
        ub = [1]*self.dof,
        cl = [-1e20]*M,
        cu = [-1e-6]*M,
        )

        #nlp.add_option('hessian_approximation', 'exact')
        nlp.add_option('sb', 'yes') # Silent Banner
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)

        if ka_0 is None:
            ka_0 = np.zeros(self.dof, dtype=np.float32)
        k_opt, self.info = nlp.solve(ka_0)                
        return g_ka * k_opt, self.info['status'], problem_obj.constraint_times
        
    def plan(self,qpos, qvel, qgoal, obs, ka_0 = None):
        # prepare the JRS
        JRS_process_time = time.perf_counter()
        _, JRS_R = zp.process_batch_JRS_trig(self.JRS_tensor,
                                             torch.as_tensor(qpos, dtype=self.dtype, device=self.device),
                                             torch.as_tensor(qvel, dtype=self.dtype, device=self.device),
                                             self.joint_axis)
        JRS_process_time = time.perf_counter() - JRS_process_time

        # Create obs zonotopes
        obs_Z = torch.cat((
            torch.as_tensor(obs[0], dtype=self.dtype, device=self.device).unsqueeze(-2),
            torch.diag_embed(torch.as_tensor(obs[1], dtype=self.dtype, device=self.device))
            ), dim=-2)
        obs_zono = zp.batchZonotope(obs_Z)

        # Compute FO
        FO_links, FO_A, FO_b, g_ka, n_obs_in_FO, FO_times = self._prepare_FO_constraints(JRS_R, obs_zono)

        # preproc_time, FO_gen_time, constraint_time = self.prepare_constraints2(env.qpos,env.qvel,env.obs_zonos)

        trajopt_time = time.perf_counter()
        k_opt, flag, constraint_times = self.trajopt(qpos, qvel, qgoal, ka_0, FO_links, FO_A, FO_b, g_ka, n_obs_in_FO)
        trajopt_time = time.perf_counter() - trajopt_time
        return k_opt, flag, trajopt_time, FO_times['constraint_gen'], FO_times['FO_gen'], JRS_process_time, constraint_times


if __name__ == '__main__':
    from zonopy.environments.urdf_obstacle import KinematicUrdfWithObstacles
    import time
    ##### 0.SET DEVICE #####
    if torch.cuda.is_available():
        device = 'cuda:0'
        #device = 'cpu'
        dtype = torch.float
    else:
        device = 'cpu'
        dtype = torch.float

    ##### LOAD ROBOT #####
    import os
    import zonopy as zp
    basedirname = os.path.dirname(zp.__file__)

    print('Loading Robot')
    # This is hardcoded for now
    import zonopy.robots2.robot as robots2
    robots2.DEBUG_VIZ = False
    rob = robots2.ArmRobot(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'))
    # rob = robots2.ArmRobot('/home/adamli/rtd-workspace/urdfs/panda_arm/panda_arm_proc.urdf')

    ##### SET ENVIRONMENT #####
    env = KinematicUrdfWithObstacles(
        robot=rob.robot,
        step_type='integration',
        check_joint_limits=True,
        check_self_collision=True,
        use_bb_collision=True,
        render_mesh=True,
        reopen_on_close=False,
        obs_size_min = [0.2,0.2,0.2],
        obs_size_max = [0.2,0.2,0.2],
        n_obs=5,
        )
    # obs = env.reset()
    obs = env.reset(
        qpos=np.array([-1.3030, -1.9067,  2.0375, -1.5399, -1.4449,  1.5094,  1.9071]),
        qvel=np.array([0,0,0,0,0,0,0.]),
        qgoal = np.array([ 0.7234,  1.6843,  2.5300, -1.0317, -3.1223,  1.2235,  1.3428]),
        obs_pos=[
            np.array([0.65,-0.46,0.33]),
            np.array([0.5,-0.43,0.3]),
            np.array([0.47,-0.45,0.15]),
            np.array([-0.3,0.2,0.23]),
            np.array([0.3,0.2,0.31])
            ])
    # obs = env.reset(
    #     qpos=np.array([ 3.1098, -0.9964, -0.2729, -2.3615,  0.2724, -1.6465, -0.5739]),
    #     qvel=np.array([0,0,0,0,0,0,0.]),
    #     qgoal = np.array([-1.9472,  1.4003, -1.3683, -1.1298,  0.7062, -1.0147, -1.1896]),
    #     obs_pos=[
    #         np.array([ 0.3925, -0.7788,  0.2958]),
    #         np.array([0.3550, 0.3895, 0.3000]),
    #         np.array([-0.0475, -0.1682, -0.7190]),
    #         np.array([0.3896, 0.5005, 0.7413]),
    #         np.array([0.4406, 0.1859, 0.1840]),
    #         np.array([ 0.1462, -0.6461,  0.7416]),
    #         np.array([-0.4969, -0.5828,  0.1111]),
    #         np.array([-0.0275,  0.1137,  0.6985]),
    #         np.array([ 0.4139, -0.1155,  0.7733]),
    #         np.array([ 0.5243, -0.7838,  0.4781])
    #         ])

    ##### 2. RUN ARMTD #####    
    planner = ARMTD_3D_planner(rob.robot)
    t_armtd = []
    T_NLP = []
    T_CONSTR = []
    T_FO = []
    T_PREPROC = []
    T_CONSTR_E = []
    N_EVALS = []
    n_steps = 100
    for _ in range(n_steps):
        ts = time.time()
        qpos, qvel, qgoal = obs['qpos'], obs['qvel'], obs['qgoal']
        obstacles = (np.asarray(obs['obstacle_pos']), np.asarray(obs['obstacle_size']))
        ka, flag, tnlp, tconstr, tfo, tpreproc, tconstraint_evals = planner.plan(qpos, qvel, qgoal, obstacles)
        t_elasped = time.time()-ts
        #print(f'Time elasped for ARMTD-3d:{t_elasped}')
        T_NLP.append(tnlp)
        T_CONSTR.append(tconstr)
        T_FO.append(tfo)
        T_PREPROC.append(tpreproc)
        T_CONSTR_E.extend(tconstraint_evals)
        N_EVALS.append(len(tconstraint_evals))
        t_armtd.append(t_elasped)
        if flag != 0:
            print("executing failsafe!")
            ka = (0 - qvel)/(T_FULL - T_PLAN)
        obs, rew, done, info = env.step(ka)
        # env.step(ka,flag)
        env.render()
    from scipy import stats
    print(f'Total time elasped for ARMTD-3D with {n_steps} steps: {stats.describe(t_armtd)}')
    print("Per step")
    print(f'NLP: {stats.describe(T_NLP)}')
    print(f'constraint evals: {stats.describe(T_CONSTR_E)}')
    print(f'number of constraint evals: {stats.describe(N_EVALS)}')
    print(f'obs buffering and constraint generation: {stats.describe(T_CONSTR)}')
    print(f'FO generation: {stats.describe(T_FO)}')
    print(f'JRS preprocessing: {stats.describe(T_PREPROC)}')