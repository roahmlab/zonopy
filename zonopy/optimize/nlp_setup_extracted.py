import torch
import numpy as np

T_PLAN, T_FULL = 0.5, 1.0
BUFFER_AMOUNT = 0.03

# NOTE
# NOTE
# NOTE: ARE YOU SOLVING K OR LAMBDA ???
class nlp_setup():
    __slots__ = [
        'qpos', 'qvel', 'g_ka', 'wrap_cont_joint_to_pi',
        'qgoal', 'M', 'dtype', 'n_timesteps', 'n_links',
        'M_obs', 'lim_flag', 'actual_pos_lim', 'vel_lim',
        'n_obs_in_frs', 'FO_link', 'A', 'b', 'dimension',
        'n_obs_cons', 'x_prev', 'Cons', 'Jac'
        ]

    def __init__(
            self,
            qpos,
            qvel,
            g_ka,
            wrap_cont_joint_to_pi,
            qgoal,
            M,
            dtype,
            n_timesteps,
            n_links,
            M_obs,
            lim_flag,
            actual_pos_lim,
            vel_lim,
            n_obs_in_frs,
            FO_link,
            A,
            b,
            dimension,
            n_obs_cons
            ):
        self.qpos = qpos
        self.qvel = qvel
        self.g_ka = g_ka
        self.wrap_cont_joint_to_pi = wrap_cont_joint_to_pi
        self.qgoal = qgoal
        self.M = M
        self.dtype = dtype
        self.n_timesteps = n_timesteps
        self.n_links = n_links
        self.M_obs = M_obs
        self.lim_flag = lim_flag
        self.actual_pos_lim = actual_pos_lim
        self.vel_lim = vel_lim
        self.n_obs_in_frs = n_obs_in_frs
        self.FO_link = FO_link
        self.A = A
        self.b = b
        self.dimension = dimension
        self.n_obs_cons = n_obs_cons

        self.x_prev = np.zeros(self.n_links)*np.nan
        self.Cons = None
        self.Jac = None
        pass

    def objective(p,x):
        qplan = p.qpos + p.qvel*T_PLAN + 0.5*p.g_ka*x*T_PLAN**2
        return torch.sum(p.wrap_cont_joint_to_pi(qplan-p.qgoal)**2)

    def gradient(p,x):
        qplan = p.qpos + p.qvel*T_PLAN + 0.5*p.g_ka*x*T_PLAN**2
        qplan_grad = 0.5*p.g_ka*T_PLAN**2
        return (2*qplan_grad*p.wrap_cont_joint_to_pi(qplan-p.qgoal)).numpy()

    def constraints(p,x): 
        p.compute_constraints(x)
        return p.Cons

    def jacobian(p,x):
        p.compute_constraints(x)
        return p.Jac

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
    def compute_constraints(p,x):
        if (p.x_prev!=x).any():                
            ka = torch.tensor(x,dtype=p.dtype).unsqueeze(0).repeat(p.n_timesteps,1)
            Cons = torch.zeros(p.M,dtype=p.dtype)
            Jac = torch.zeros(p.M,p.n_links,dtype=p.dtype)
            # Hess = torch.zeros(M,self.n_links,self.n_links,dtype=self.dtype)
            
            # position and velocity constraints
            t_peak_optimum = -p.qvel/(p.g_ka*ka[0]) # time to optimum of first half traj.
            qpos_peak_optimum = (t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(p.qpos+p.qvel*t_peak_optimum+0.5*(p.g_ka*ka[0])*t_peak_optimum**2).nan_to_num()
            #grad_qpos_peak_optimum = torch.diag((t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(0.5*self.g_ka*t_peak_optimum**2).nan_to_num())
            grad_qpos_peak_optimum = torch.diag((t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(0.5*p.qvel**2/(p.g_ka*ka[0]**2)).nan_to_num())
            # hess_qpos_peak_optimum = torch.sparse_coo_tensor([list(range(self.n_links))]*3,(t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(-self.qvel**2/(self.g_ka*ka[0]**3)).nan_to_num(),(self.n_links,)*3).to_dense()

            qpos_peak = p.qpos + p.qvel * T_PLAN + 0.5 * (p.g_ka * ka[0]) * T_PLAN**2
            grad_qpos_peak = 0.5 * p.g_ka * T_PLAN**2 * torch.eye(p.n_links,dtype=p.dtype)
            qvel_peak = p.qvel + p.g_ka * ka[0] * T_PLAN
            grad_qvel_peak = p.g_ka * T_PLAN * torch.eye(p.n_links,dtype=p.dtype)

            bracking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
            qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*bracking_accel*(T_FULL-T_PLAN)**2
            # can be also, qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL+T_PLAN) + 0.5 * (self.g_ka * ka[0]) * T_PLAN * T_FULL
            grad_qpos_brake = 0.5 * p.g_ka * T_PLAN * T_FULL * torch.eye(p.n_links,dtype=p.dtype) # NOTE: need to verify equation

            qpos_possible_max_min = torch.vstack((qpos_peak_optimum,qpos_peak,qpos_brake))[:,p.lim_flag] 
            qpos_ub = (qpos_possible_max_min - p.actual_pos_lim[:,0]).flatten()
            qpos_lb = (p.actual_pos_lim[:,1] - qpos_possible_max_min).flatten()
            
            grad_qpos_ub = torch.vstack((grad_qpos_peak_optimum[p.lim_flag],grad_qpos_peak[p.lim_flag],grad_qpos_brake[p.lim_flag]))
            grad_qpos_lb = - grad_qpos_ub

            Cons[p.M_obs:] = torch.hstack((qvel_peak-p.vel_lim, -p.vel_lim-qvel_peak,qpos_ub,qpos_lb))
            Jac[p.M_obs:] = torch.vstack((grad_qvel_peak, -grad_qvel_peak, grad_qpos_ub, grad_qpos_lb))
            #Hess[M_obs+2*self.n_links:M_obs+2*self.n_links+self.n_pos_lim] = hess_qpos_peak_optimum[self.lim_flag]
            #Hess[M_obs+2*self.n_links+3*self.n_pos_lim:M_obs+2*self.n_links+4*self.n_pos_lim] = - hess_qpos_peak_optimum[self.lim_flag]

            if p.n_obs_in_frs > 0:
                for j in range(p.n_links):
                    c_k = p.FO_link[j].center_slice_all_dep(ka)
                    grad_c_k = p.FO_link[j].grad_center_slice_all_dep(ka)
                    # hess_c_k = self.FO_link[j].hess_center_slice_all_dep(ka)
                    h_obs = (p.A[j]@c_k.unsqueeze(-1)).squeeze(-1) - p.b[j]
                    cons_obs, ind = torch.max(h_obs.nan_to_num(-torch.inf),-1)
                    A_max = p.A[j].gather(-2,ind.reshape(p.n_obs_in_frs,p.n_timesteps,1,1).repeat(1,1,1,p.dimension))
                    grad_obs = (A_max@grad_c_k).reshape(p.n_obs_cons,p.n_links)
                    Cons[j*p.n_obs_cons:(j+1)*p.n_obs_cons] = - cons_obs.reshape(p.n_obs_cons)
                    Jac[j*p.n_obs_cons:(j+1)*p.n_obs_cons] = - grad_obs
                    # Hess[j*n_obs_cons:(j+1)*n_obs_cons] = - hess_obs
                    
            
            p.Cons = Cons.numpy()
            p.Jac = Jac.numpy()
            # p.Hess = Hess.numpy()
            p.x_prev = np.copy(x)   

    def intermediate(p, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                d_norm, regularization_size, alpha_du, alpha_pr,
                ls_trials):
        pass