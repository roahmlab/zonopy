import torch
import numpy as np
import time

class OfflineArmtdFoConstraints:
    def __init__(self, dimension = 3, dtype = torch.float):
        self.dimension = dimension
        self.dtype = dtype
        self.np_dtype = torch.empty(0,dtype=dtype).numpy().dtype
    
    def set_params(self, FO_link_zono, A, b, g_ka, n_obs_in_FO, n_joints):
        self.FO_link_zono = FO_link_zono
        self.A = A
        self.b = b
        self.g_ka = g_ka
        self.n_obs_in_FO = n_obs_in_FO
        self.n_links = len(FO_link_zono)
        self.n_timesteps = FO_link_zono[0].batch_shape[0]
        self.n_obs_cons = self.n_timesteps * n_obs_in_FO
        self.M = self.n_links * self.n_obs_cons
        self.n_joints = n_joints

    def __call__(self, x, Cons_out=None, Jac_out=None):
        x = torch.as_tensor(x, dtype=self.dtype)
        if Cons_out is None:
            Cons_out = np.empty(self.M, dtype=self.np_dtype)
        if Jac_out is None:
            Jac_out = np.empty((self.M, self.n_joints), dtype=self.np_dtype)

        for j in range(self.n_links):
            # slice the center and get the gradient for it
            c_k = self.FO_link_zono[j].center_slice_all_dep(x)
            grad_c_k = self.FO_link_zono[j].grad_center_slice_all_dep(x)

            # use those to compute the halfspace constraints
            h_obs = (self.A[j]@c_k.unsqueeze(-1)).squeeze(-1) - self.b[j]
            cons_obs, ind = torch.max(h_obs.nan_to_num(-torch.inf),-1)
            ind = ind.reshape(self.n_obs_in_FO,self.n_timesteps,1,1).expand(-1,-1,-1,self.dimension)
            A_max = self.A[j].gather(-2, ind)
            grad_obs = (A_max@grad_c_k).reshape(self.n_obs_cons, self.n_joints)

            Cons_out[j*self.n_obs_cons:(j+1)*self.n_obs_cons] = -cons_obs.reshape(self.n_obs_cons).numpy()
            Jac_out[j*self.n_obs_cons:(j+1)*self.n_obs_cons] = -grad_obs.numpy()
        
        return Cons_out, Jac_out

# NOTE
# NOTE
# NOTE: ARE YOU SOLVING K OR LAMBDA ???
class ArmtdNlpProblem():
    __slots__ = [
        't_plan',
        't_full',
        'dtype',
        'n_joints',
        'g_ka',
        'pos_lim',
        'vel_lim',
        'continuous_joints',
        'pos_lim_mask',
        'M_limits',
        '_g_ka_masked',
        '_pos_lim_masked',
        '_masked_eye',
        '_grad_qpos_peak',
        '_grad_qvel_peak',
        '_grad_qpos_brake',
        'qpos',
        'qvel',
        'qgoal',
        '_FO_constraint',
        'M',
        '_qpos_masked',
        '_qvel_masked',
        '_Cons',
        '_Jac',
        '_x_prev',
        '_constraint_times']

    def __init__(
            self,
            n_joints: int,
            g_ka: np.ndarray, #[object],
            pos_lim: np.ndarray, #[float],
            vel_lim: np.ndarray, #[float],
            continuous_joints: np.ndarray, #[int],
            pos_lim_mask: np.ndarray, #[bool],
            dtype: torch.dtype = torch.float,
            t_plan: float = 0.5,
            t_full: float = 1.0
            ):

        # Core constants
        self.t_plan = t_plan
        self.t_full = t_full
        # convert from torch dtype to np dtype
        self.dtype = torch.empty(0,dtype=dtype).numpy().dtype

        # Optimization parameters and range, which is known a priori for armtd
        self.n_joints = n_joints
        self.g_ka = g_ka

        # Joint limit constraints
        self.pos_lim = pos_lim
        self.vel_lim = vel_lim
        self.continuous_joints = continuous_joints
        self.pos_lim_mask = pos_lim_mask
        self.M_limits = int(2*self.n_joints + 6*self.pos_lim_mask.sum())

        # Extra usefuls precomputed for joint limit
        self._g_ka_masked = self.g_ka[self.pos_lim_mask]
        self._pos_lim_masked = self.pos_lim[:,self.pos_lim_mask]
        self._masked_eye = np.eye(self.n_joints, dtype=self.dtype)[self.pos_lim_mask]

        # Precompute some joint limit gradients
        self._grad_qpos_peak = (0.5 * self._g_ka_masked * self.t_plan**2).reshape(-1,1) * self._masked_eye
        self._grad_qvel_peak = np.diag(self.g_ka * self.t_plan)
        self._grad_qpos_brake = (0.5 * self._g_ka_masked * self.t_plan * self.t_full).reshape(-1,1) * self._masked_eye

    def reset(self, qpos, qvel, qgoal, FO_constraint):
        # Key parameters for optimization
        self.qpos = qpos
        self.qvel = qvel
        self.qgoal = qgoal
        # FO constraint holder & functional
        self._FO_constraint = FO_constraint

        self.M = self.M_limits
        if FO_constraint is not None:
            self.M += int(FO_constraint.M)
        
        # Prepare the rest of the joint limit constraints
        self._qpos_masked = self.qpos[self.pos_lim_mask]
        self._qvel_masked = self.qvel[self.pos_lim_mask]

        # Constraints and Jacobians
        self._Cons = np.zeros(self.M, dtype=self.dtype)
        self._Jac = np.zeros((self.M, self.n_joints), dtype=self.dtype)

        # Internal
        self._x_prev = np.zeros(self.n_joints)*np.nan
        self._constraint_times = []

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

    def compute_constraints(self,x):
        if (self._x_prev!=x).any():
            start = time.perf_counter()
            self._x_prev = np.copy(x)

            # zero out the underlying constraints and jacobians
            self._Cons[...] = 0
            self._Jac[...] = 0

            # Joint limits
            self._constraints_limits(x, Cons_out=self._Cons[:self.M_limits], Jac_out=self._Jac[:self.M_limits])

            # FO if needed
            if self._FO_constraint.n_obs_in_FO > 0:
                self._FO_constraint(x, Cons_out=self._Cons[self.M_limits:], Jac_out=self._Jac[self.M_limits:])
            
            # Timing
            self._constraint_times.append(time.perf_counter() - start)

    def _constraints_limits(self, x, Cons_out=None, Jac_out=None):
        ka = x # is this numpy? we will see
        if Cons_out is None:
            Cons_out = np.empty(self.M_limits, dtype=self.dtype)
        if Jac_out is None:
            Jac_out = np.empty((self.M_limits, self.n_joints), dtype=self.dtype)

        ## position and velocity constraints
        # disable warnings for this section
        settings = np.seterr('ignore')
        # scake k and get the part relevant to the constrained positions
        scaled_k = self.g_ka*ka
        scaled_k_masked = scaled_k[self.pos_lim_mask]
        # time to optimum of first half traj.
        t_peak_optimum = -self._qvel_masked/scaled_k_masked
        # if t_peak_optimum is in the time, qpos_peak_optimum has a value
        t_peak_in_range = (t_peak_optimum > 0) * (t_peak_optimum < self.t_plan)
        # Get the position and gradient at the peak
        qpos_peak_optimum = np.nan_to_num(t_peak_in_range * (self._qpos_masked + self._qvel_masked * t_peak_optimum + 0.5 * scaled_k_masked * t_peak_optimum**2))
        grad_qpos_peak_optimum = np.nan_to_num(t_peak_in_range * 0.5*self._qvel_masked**2/(scaled_k_masked**2)).reshape(-1,1) * self._masked_eye
        # restore
        np.seterr(**settings)

        ## Position and velocity at velocity peak of trajectory
        qpos_peak = self._qpos_masked + self._qvel_masked * self.t_plan + 0.5 * scaled_k_masked * self.t_plan**2
        qvel_peak = self.qvel + scaled_k * self.t_plan

        ## Position at braking
        # braking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
        # qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*braking_accel*(T_FULL-T_PLAN)**2
        # NOTE: swapped to simplified form
        qpos_brake = self._qpos_masked + 0.5 * self._qvel_masked * (self.t_full + self.t_plan) + 0.5 * scaled_k_masked * self.t_plan * self.t_full
        
        ## compute the final constraint values and store them to the desired output arrays
        qpos_possible_max_min = np.vstack((qpos_peak_optimum,qpos_peak,qpos_brake))
        qpos_ub = (qpos_possible_max_min - self._pos_lim_masked[1]).flatten()
        qpos_lb = (self._pos_lim_masked[0] - qpos_possible_max_min).flatten()
        qvel_ub = qvel_peak - self.vel_lim
        qvel_lb = (-self.vel_lim) - qvel_peak
        np.concatenate((qpos_ub, qpos_lb, qvel_ub, qvel_lb), out=Cons_out)

        ## Do the same for the gradients
        grad_qpos_ub = np.vstack((grad_qpos_peak_optimum,self._grad_qpos_peak,self._grad_qpos_brake))
        grad_qpos_lb = -grad_qpos_ub
        grad_qvel_ub = self._grad_qvel_peak
        grad_qvel_lb = -self._grad_qvel_peak
        np.concatenate((grad_qpos_ub, grad_qpos_lb, grad_qvel_ub, grad_qvel_lb), out=Jac_out)

        return Cons_out, Jac_out
    
    def _wrap_cont_joints(self, pos: np.ndarray) -> np.ndarray:
        pos = np.copy(pos)
        pos[..., self.continuous_joints] = (pos[..., self.continuous_joints] + np.pi) % (2 * np.pi) - np.pi
        return pos

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                d_norm, regularization_size, alpha_du, alpha_pr,
                ls_trials):
        pass

    @property
    def constraint_times(self):
        return self._constraint_times