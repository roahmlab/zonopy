import torch
import numpy as np
import time

# NOTE
# NOTE
# NOTE: ARE YOU SOLVING K OR LAMBDA ???
class armtd_nlp():
    __slots__ = [
        'dimension',
        'n_obs_cons',
        'qpos',
        'qvel',
        'qgoal',
        'FO_link_zono',
        'A',
        'b',
        'g_ka',
        'dtype',
        'n_links',
        'n_joints',
        'n_frs_timesteps',
        'n_obs_in_FO',
        'M',
        'M_fo',
        'M_limits',
        't_plan',
        't_full',
        'pos_lim',
        'vel_lim',
        'continuous_joints',
        'pos_lim_mask',
        '_g_ka_masked',
        '_qpos_masked',
        '_qvel_masked',
        '_pos_lim_masked',
        '_masked_eye',
        '_grad_qpos_peak',
        '_grad_qvel_peak',
        '_grad_qpos_brake',
        '_Cons',
        '_Jac',
        '_x_prev',
        '_constraint_times']

    def __init__(
            self,
            qpos: np.ndarray, #[float],
            qvel: np.ndarray, #[float],
            qgoal: np.ndarray, #[float],
            FO_link_zono: np.ndarray, #[object],
            A: np.ndarray, #[torch.tensor],
            b: np.ndarray, #[torch.tensor],
            g_ka: np.ndarray, #[object],
            dtype: np.dtype,
            n_links: int,
            n_joints: int,
            n_frs_timesteps: int,
            n_obs_in_FO: int,
            M: int,
            M_fo: int,
            M_limits: int,
            pos_lim: np.ndarray, #[float],
            vel_lim: np.ndarray, #[float],
            continuous_joints: np.ndarray, #[int],
            pos_lim_mask: np.ndarray, #[bool],
            dimension: int, # del?
            n_obs_cons: int, # del?
            t_plan: float = 0.5,
            t_full: float = 1.0
            ):
        # self.qpos = qpos
        # self.qvel = qvel
        # self.g_ka = g_ka
        # self.wrap_cont_joint_to_pi = wrap_cont_joint_to_pi
        # self.qgoal = qgoal
        # self.M = M
        # self.dtype = dtype
        # self.n_timesteps = n_timesteps
        # self.n_links = n_links
        # self.M_obs = M_obs
        # self.lim_flag = lim_flag
        # self.actual_pos_lim = actual_pos_lim
        # self.vel_lim = vel_lim
        # self.n_obs_in_frs = n_obs_in_frs
        # self.FO_link = FO_link
        # self.A = A
        # self.b = b
        self.dimension = dimension
        self.n_obs_cons = n_obs_cons

        self.qpos = qpos
        self.qvel = qvel
        self.qgoal = qgoal
        self.FO_link_zono = FO_link_zono
        self.A = A
        self.b = b
        self.g_ka = g_ka
        self.dtype = dtype

        self.n_links = n_links
        self.n_joints = n_joints
        self.n_frs_timesteps = n_frs_timesteps
        self.n_obs_in_FO = n_obs_in_FO
        self.M = M
        self.M_fo = M_fo
        self.M_limits = M_limits
        self.t_plan = t_plan
        self.t_full = t_full

        # NEW
        self.pos_lim = pos_lim
        self.vel_lim = vel_lim
        self.continuous_joints = continuous_joints
        self.pos_lim_mask = pos_lim_mask

        # Extra usefuls precomputed for joint limit
        self._g_ka_masked = self.g_ka[self.pos_lim_mask]
        self._qpos_masked = self.qpos[self.pos_lim_mask]
        self._qvel_masked = self.qvel[self.pos_lim_mask]
        self._pos_lim_masked = pos_lim[:,self.pos_lim_mask]
        self._masked_eye = np.eye(self.n_joints, dtype=self.dtype)[self.pos_lim_mask]

        # Precompute some joint limit gradients
        self._grad_qpos_peak = (0.5 * self._g_ka_masked * self.t_plan**2).reshape(-1,1) * self._masked_eye
        self._grad_qvel_peak = np.diag(self.g_ka * self.t_plan)
        self._grad_qpos_brake = (0.5 * self._g_ka_masked * self.t_plan * self.t_full).reshape(-1,1) * self._masked_eye

        # Constraints and Jacobians
        self._Cons = np.zeros(self.M, dtype=self.dtype)
        self._Jac = np.zeros((self.M, self.n_joints), dtype=self.dtype)

        # Internal
        self._x_prev = np.zeros(self.n_joints)*np.nan
        self._constraint_times = []
        pass

    def objective(self, x):
        qplan = self.qpos + self.qvel*self.t_plan + 0.5*self.g_ka*x*self.t_plan**2
        return np.sum(self._wrap_cont_joints(qplan-self.qgoal)**2)

    def gradient(self, x):
        qplan = self.qpos + self.qvel*self.t_plan + 0.5*self.g_ka*x*self.t_plan**2
        qplan_grad = 0.5*self.g_ka*self.t_plan**2
        return 2*qplan_grad*self._wrap_cont_joints(qplan-self.qgoal)

    def constraints(self, x): 
        self.compute_constraints(x)
        return self._Cons

    def jacobian(self, x):
        self.compute_constraints(x)
        return self._Jac

    '''
    def hessianstructure(p):
        return np.nonzero(np.tril(np.ones((self.n_links,self.n_links))))

    def hessian(p, x, lagrange, obj_factor):
        p.compute_constraints(x)
        H = obj_factor*0.5*self.g_ka**2*T_PLAN**4*np.eye(self.n_links) + np.sum(lagrange.reshape(-1,1,1)*p.Hess,0)
        row, col = p.hessianstructure()
        return H[row,col]
        #return H 
    '''

    def _compute_limit_constraints(self, x, Cons_out=None, Jac_out=None):
        ka = x # is this numpy? we will see
        if Cons_out is None:
            Cons_out = np.empty(self.M_limits, dtype=self.dtype)
        if Jac_out is None:
            Jac_out = np.empty((self.M_limits, self.n_joints), dtype=self.dtype)

        ## position and velocity constraints
        scaled_k = self.g_ka*ka
        scaled_k_masked = scaled_k[self.pos_lim_mask]
        # time to optimum of first half traj.
        # t_peak_optimum = -self.qvel/scaled_k
        t_peak_optimum = -self._qvel_masked/scaled_k_masked
        # if t_peak_optimum is in the time, qpos_peak_optimum has a value
        t_peak_in_range = (t_peak_optimum > 0) * (t_peak_optimum < self.t_plan)
        # qpos_peak_optimum = t_peak_in_range * (self.qpos + self.qvel * t_peak_optimum + 0.5 * scaled_k * t_peak_optimum**2).nan_to_num()
        qpos_peak_optimum = t_peak_in_range * np.nan_to_num(self._qpos_masked + self._qvel_masked * t_peak_optimum + 0.5 * scaled_k_masked * t_peak_optimum**2)
        #grad_qpos_peak_optimum = torch.diag((t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(0.5*self.g_ka*t_peak_optimum**2).nan_to_num())
        # grad_qpos_peak_optimum = torch.diag(t_peak_in_range * (0.5*self.qvel**2/(scaled_k**2)).nan_to_num())
        grad_qpos_peak_optimum = (t_peak_in_range * np.nan_to_num(0.5*self._qvel_masked**2/(scaled_k_masked**2))).reshape(-1,1) * self._masked_eye
        # hess_qpos_peak_optimum = torch.sparse_coo_tensor([list(range(self.n_links))]*3,(t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(-self.qvel**2/(scaled_k**3)).nan_to_num(),(self.n_links,)*3).to_dense()

        # Position and velocity at velocity peak of trajectory
        # qpos_peak = self.qpos + self.qvel * T_PLAN + 0.5 * scaled_k * T_PLAN**2
        # grad_qpos_peak = torch.diag(0.5 * self.g_ka * T_PLAN**2, dtype=p.dtype)
        qpos_peak = self._qpos_masked + self._qvel_masked * self.t_plan + 0.5 * scaled_k_masked * self.t_plan**2
        qvel_peak = self.qvel + scaled_k * self.t_plan

        # Position at braking
        # braking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
        # qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*braking_accel*(T_FULL-T_PLAN)**2
        # NOTE: swapped to simplified form
        # qpos_brake = self.qpos + 0.5 * self.qvel * (T_FULL + T_PLAN) + 0.5 * scaled_k * T_PLAN * T_FULL
        # grad_qpos_brake = torch.diag(0.5 * self.g_ka * T_PLAN * T_FULL, dtype=p.dtype)
        qpos_brake = self._qpos_masked + 0.5 * self._qvel_masked * (self.t_full + self.t_plan) + 0.5 * scaled_k_masked * self.t_plan * self.t_full
        


        # qpos_possible_max_min = torch.vstack((qpos_peak_optimum,qpos_peak,qpos_brake))[:,p.lim_flag] 
        # qpos_ub = (qpos_possible_max_min - p.actual_pos_lim[:,0]).flatten()
        # qpos_lb = (p.actual_pos_lim[:,1] - qpos_possible_max_min).flatten()
        qpos_possible_max_min = np.vstack((qpos_peak_optimum,qpos_peak,qpos_brake))
        qpos_ub = (qpos_possible_max_min - self._pos_lim_masked[1]).flatten()
        qpos_lb = (self._pos_lim_masked[0] - qpos_possible_max_min).flatten()
        qvel_ub = qvel_peak - self.vel_lim
        qvel_lb = (-self.vel_lim) - qvel_peak
        np.concatenate((qpos_ub, qpos_lb, qvel_ub, qvel_lb), out=Cons_out)

        grad_qpos_ub = np.vstack((grad_qpos_peak_optimum,self._grad_qpos_peak,self._grad_qpos_brake))
        grad_qpos_lb = -grad_qpos_ub
        grad_qvel_ub = self._grad_qvel_peak
        grad_qvel_lb = -self._grad_qvel_peak
        np.concatenate((grad_qpos_ub, grad_qpos_lb, grad_qvel_ub, grad_qvel_lb), out=Jac_out)

        # Cons[p.M_obs:] = torch.hstack((qvel_peak-p.vel_lim, -p.vel_lim-qvel_peak,qpos_ub,qpos_lb))
        # Jac[p.M_obs:] = torch.vstack((grad_qvel_peak, -grad_qvel_peak, grad_qpos_ub, grad_qpos_lb))
        #Hess[M_obs+2*self.n_links:M_obs+2*self.n_links+self.n_pos_lim] = hess_qpos_peak_optimum[self.lim_flag]
        #Hess[M_obs+2*self.n_links+3*self.n_pos_lim:M_obs+2*self.n_links+4*self.n_pos_lim] = - hess_qpos_peak_optimum[self.lim_flag]

        return Cons_out, Jac_out
    
    def _FO_constraint(self, x, Cons_out=None, Jac_out=None):
        # ka = torch.as_tensor(x, dtype=torch.float).expand(self.n_frs_timesteps,-1)
        x = torch.as_tensor(x, dtype=torch.float)
        if Cons_out is None:
            Cons_out = np.empty(self.M_fo, dtype=self.dtype)
        if Jac_out is None:
            Jac_out = np.empty((self.M_fo, self.n_joints), dtype=self.dtype)

        for j in range(self.n_links):
            # c_k = self.FO_link_zono[j].center_slice_all_dep(ka)
            # grad_c_k = self.FO_link_zono[j].grad_center_slice_all_dep(ka)
            c_k = self.FO_link_zono[j].center_slice_all_dep(x)
            grad_c_k = self.FO_link_zono[j].grad_center_slice_all_dep(x)
            # c_k_comp = self.FO_link_zono[j].center_slice_all_dep(x)
            # grad_c_k_comp = self.FO_link_zono[j].grad_center_slice_all_dep(x)
            # assert((c_k == c_k_comp).all())
            # if not (grad_c_k == grad_c_k_comp).all():
            #     c_k_comp = self.FO_link_zono[j].center_slice_all_dep(x)
            #     grad_c_k_comp = self.FO_link_zono[j].grad_center_slice_all_dep(x)
            #     print('ACK')
            

            h_obs = (self.A[j]@c_k.unsqueeze(-1)).squeeze(-1) - self.b[j]

            cons_obs, ind = torch.max(h_obs.nan_to_num(-torch.inf),-1)
            # A_max_test = self.A[j][..., ind, :]
            ind = ind.reshape(self.n_obs_in_FO,self.n_frs_timesteps,1,1).expand(-1,-1,-1,self.dimension)
            A_max = self.A[j].gather(-2, ind)
            # assert((A_max == A_max_test).all())
            grad_obs = (A_max@grad_c_k).reshape(self.n_obs_cons, self.n_joints)

            Cons_out[j*self.n_obs_cons:(j+1)*self.n_obs_cons] = -cons_obs.reshape(self.n_obs_cons).numpy()
            Jac_out[j*self.n_obs_cons:(j+1)*self.n_obs_cons] = -grad_obs.numpy()
        
        return Cons_out, Jac_out

    def compute_constraints(self,x):
        if (self._x_prev!=x).any():
            start = time.perf_counter()
            self._x_prev = np.copy(x)

            # zero out the underlying constraints and jacobians
            self._Cons[...] = 0
            self._Jac[...] = 0

            # Joint limits
            self._compute_limit_constraints(x, Cons_out=self._Cons[self.M_fo:], Jac_out=self._Jac[self.M_fo:])

            # FO if needed
            if self.n_obs_in_FO > 0:
                self._FO_constraint(x, Cons_out=self._Cons[:self.M_fo], Jac_out=self._Jac[:self.M_fo])
            
            # Timing
            self._constraint_times.append(time.perf_counter() - start)

    def _wrap_cont_joints(self, pos: np.ndarray) -> np.ndarray:
        pos = np.copy(pos)
        pos[..., self.continuous_joints] = (pos[..., self.continuous_joints] + np.pi) % (2 * np.pi) - np.pi
        return pos

    def intermediate(p, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                d_norm, regularization_size, alpha_du, alpha_pr,
                ls_trials):
        pass

    @property
    def constraint_times(self):
        return self._constraint_times