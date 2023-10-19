# TODO VALIDATE

import torch
import numpy as np
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy as forward_occupancy
import cyipopt


import time

T_PLAN, T_FULL = 0.5, 1.0

from typing import List
from zonopy.optimize.armtd_nlp_problem import ArmtdNlpProblem, OfflineArmtdFoConstraints
from zonopy.robots2.robot import ZonoArmRobot

class ARMTD_3D_planner():
    def __init__(self,
                 robot: ZonoArmRobot,
                 zono_order: int = 2, # this appears to have been 40 before but it was ignored for 2
                 max_combs: int = 200,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
                 include_end_effector: bool = False,
                 ):
        self.dtype, self.device = dtype, device
        self.np_dtype = torch.empty(0,dtype=dtype).numpy().dtype

        self.robot = robot
        self.PI = torch.tensor(torch.pi,dtype=self.dtype,device=self.device)
        self.JRS_tensor = zp.preload_batch_JRS_trig(dtype=self.dtype,device=self.device)
        
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.combs = self._generate_combinations_upto(max_combs)
        self.include_end_effector = include_end_effector

        self._setup_robot(robot)

        # Prepare the nlp
        self.g_ka = np.ones((self.dof),dtype=self.np_dtype) * np.pi/24                  # Hardcoded because it's preloaded...
        self.nlp_problem_obj = ArmtdNlpProblem(self.dof,
                                         self.g_ka,
                                         self.pos_lim, 
                                         self.vel_lim,
                                         self.continuous_joints,
                                         self.pos_lim_mask,
                                         self.dtype,
                                         T_PLAN,
                                         T_FULL)

        # self.wrap_env(env)
        # self.n_timesteps = 100
        #self.joint_speed_limit = torch.vstack((torch.pi*torch.ones(n_links),-torch.pi*torch.ones(n_links)))
    
    def _setup_robot(self, robot: ZonoArmRobot):
        self.dof = robot.dof
        self.joint_axis = robot.joint_axis
        self.pos_lim = robot.np.pos_lim
        self.vel_lim = robot.np.vel_lim
        # self.eff_lim = np.array(eff_lim) # Unused for now
        self.continuous_joints = robot.np.continuous_joints
        self.pos_lim_mask = robot.np.pos_lim_mask
        pass

    def _generate_combinations_upto(self, max_combs):
        return [torch.combinations(torch.arange(i,device=self.device),2) for i in range(max_combs+1)]
        
    def _prepare_FO_constraints(self,
                                JRS_R: zp.batchMatPolyZonotope,
                                obs_zono: zp.batchZonotope,
                                ):
        # constant
        n_obs = len(obs_zono)

        ### get the forward occupancy
        FO_gen_time = time.perf_counter()
        FO_links, _ = forward_occupancy(JRS_R, self.robot, self.zono_order)
        # let's assume we can ignore the base link and convert to a list of pz's
        # end effector link is a thing???? (n actuated_joints = 7, n links = 8)
        FO_links = list(FO_links.values())[1:]
        if not self.include_end_effector:
            FO_links = FO_links[:-1]
        # two more constants
        n_links = len(FO_links)
        n_frs_timesteps = FO_links[0].batch_shape[0]

        ### begin generating constraints
        constraint_gen_time = time.perf_counter()
        out_A = np.empty((n_links),dtype=object)
        out_b = np.empty((n_links),dtype=object)
        out_g_ka = self.g_ka
        out_FO_links = np.empty((n_links),dtype=object)

        # Get the obstacle Z matrix
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
        FO_constraint = OfflineArmtdFoConstraints(dtype=dtype)
        FO_constraint.set_params(out_FO_links, out_A, out_b, out_g_ka, out_n_obs_in_frs, self.dof)
        return FO_constraint, out_times

    def trajopt(self, qpos, qvel, qgoal, ka_0, FO_constraint):
        # Moved to another file
        self.nlp_problem_obj.reset(qpos, qvel, qgoal, FO_constraint)
        n_constraints = self.nlp_problem_obj.M

        nlp = cyipopt.Problem(
        n = self.dof,
        m = n_constraints,
        problem_obj=self.nlp_problem_obj,
        lb = [-1]*self.dof,
        ub = [1]*self.dof,
        cl = [-1e20]*n_constraints,
        cu = [-1e-6]*n_constraints,
        )

        #nlp.add_option('hessian_approximation', 'exact')
        nlp.add_option('sb', 'yes') # Silent Banner
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)

        if ka_0 is None:
            ka_0 = np.zeros(self.dof, dtype=np.float32)
        k_opt, self.info = nlp.solve(ka_0)                
        return FO_constraint.g_ka * k_opt, self.info['status'], self.nlp_problem_obj.constraint_times
        
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
            torch.diag_embed(torch.as_tensor(obs[1], dtype=self.dtype, device=self.device))/2.
            ), dim=-2)
        obs_zono = zp.batchZonotope(obs_Z)

        # Compute FO
        FO_constraint, FO_times = self._prepare_FO_constraints(JRS_R, obs_zono)

        # preproc_time, FO_gen_time, constraint_time = self.prepare_constraints2(env.qpos,env.qvel,env.obs_zonos)

        trajopt_time = time.perf_counter()
        k_opt, flag, constraint_times = self.trajopt(qpos, qvel, qgoal, ka_0, FO_constraint)
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
    rob = robots2.ZonoArmRobot.load(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'), device=device, dtype=dtype)
    # rob = robots2.ArmRobot('/home/adamli/rtd-workspace/urdfs/panda_arm/panda_arm_proc.urdf')

    ##### SET ENVIRONMENT #####
    env = KinematicUrdfWithObstacles(
        robot=rob.urdf,
        step_type='integration',
        check_joint_limits=True,
        check_self_collision=False,
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
    planner = ARMTD_3D_planner(rob, device=device)
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
        assert(not info['collision_info']['in_collision'])
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