import numpy as np 
import torch 

T_PLAN, T_FULL = 0.5, 1.0
def wrap_to_pi(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi

def wrap_cont_joint_to_pi(phases,lim_flag):
    phases_new = np.copy(phases)
    phases_new[~lim_flag] = (phases[~lim_flag] + torch.pi) % (2 * torch.pi) - torch.pi
    return phases_new

class NlpSetup():
    def __init__(self,A,b,FO_link,qpos,qvel,qgoal,n_timesteps,n_links,n_obs,dimension,g_ka):
        self.qpos = qpos
        self.qvel = qvel 
        self.qgoal = qgoal 
        self.x_prev = np.zeros(n_links)*np.nan
        self.n_timesteps = n_timesteps
        self.n_links = n_links 
        self.n_obs = n_obs 
        self.M_obs = self.n_timesteps*self.n_links*self.n_obs
        self.M = self.M_obs + 2*self.n_links
        self.dimension = dimension
        self.g_ka = g_ka
        self.parse_FO_link(FO_link)
        self.A = A 
        self.b = b

    def parse_FO_link(self,FO_link):
        self.c, self.G, self.n_ids, self.expMat, self.expMat_red = [],[],[],[],[]
        for j in range(self.n_links):
            pz = FO_link[j]
            self.c.append(pz.c.cpu().numpy())
            self.G.append(pz.G.cpu().numpy())
            n_ids = len(pz.id)
            expMat = pz.expMat[:,torch.argsort(pz.id)].cpu().numpy()
            expMat_red = np.expand_dims(expMat,0).repeat(n_ids,axis=0) - np.expand_dims(np.eye(n_ids),-2)
            self.n_ids.append(n_ids) 
            self.expMat.append(expMat)
            self.expMat_red.append(expMat_red.clip(min=0)) # dont let this be smaller than 0
        self.batch_idx_all = pz.batch_idx_all
    
    def FO_center_slice_all_dep(self,val_slc,j):
        val_slc = np.expand_dims(val_slc[:self.n_ids[j]],-2) # numpy take
        return self.c[j] + (np.expand_dims(np.prod(val_slc**self.expMat[j],axis=-1),-2)@self.G[j]).squeeze(-2)
    
    def FO_grad_center_slice_all_dep(self,val_slc,j):
        n_short = val_slc.shape[-1]-self.n_ids[j]
        val_slc = np.expand_dims(val_slc[:self.n_ids[j]],(-2,-3))
        grad = (self.expMat[j].T*np.nan_to_num(np.prod(val_slc**self.expMat_red[j],axis=-1),0)@self.G[j]).transpose(tuple(range(len(self.batch_idx_all)))+(-1,-2))    
        return np.concatenate((grad,np.zeros((grad.shape[:-1]+(n_short,)))),axis=-1)

    def objective(self,x):
        qplan = self.qpos + self.qvel*T_PLAN + 0.5*self.g_ka*x*T_PLAN**2
        return np.sum(wrap_to_pi(qplan-self.qgoal)**2)

    def gradient(self,x):
        qplan = self.qpos + self.qvel*T_PLAN + 0.5*self.g_ka*x*T_PLAN**2
        return self.g_ka* T_PLAN**2*wrap_to_pi(qplan - self.qgoal)

    def constraints(self,x): 
        self.compute_constraints_jacobian(x)
        return self.cons

    def jacobian(self,x):
        self.compute_constraints_jacobian(x)
        return self.jac
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                d_norm, regularization_size, alpha_du, alpha_pr,
                ls_trials):
        pass


class NlpSetup2D(NlpSetup):
    def compute_constraints_jacobian(self,x):
        #ka = np.expand_dims(x,0).repeat(self.n_timesteps,axis=0)
        if (self.x_prev!=x).any():
            self.cons = np.zeros((self.M))
            self.jac = np.zeros((self.M,self.n_links))
            # velocity min max constraints
            q_peak = self.qvel + self.g_ka * x * T_PLAN
            grad_q_peak = self.g_ka * T_PLAN * np.eye(self.n_links)
            self.cons[self.M_obs:] = np.hstack((q_peak-np.pi, -np.pi-q_peak))
            self.jac[self.M_obs:] = np.vstack((grad_q_peak, -grad_q_peak))
            for j in range(self.n_links):
                c_k = np.expand_dims(self.FO_center_slice_all_dep(x,j),-1)
                grad_c_k = self.FO_grad_center_slice_all_dep(x,j)
                for o in range(self.n_obs):
                    h_obs = (self.A[j][o]@c_k).squeeze(-1) - self.b[j][o]
                    ind = np.argmax(np.nan_to_num(h_obs,-np.inf),-1) 
                    cons = - np.take_along_axis(h_obs,ind.reshape(self.n_timesteps,1),axis=1).squeeze(-1) # shape: n_timsteps, SAFE if <=-1e-6
                    jac = - (np.take_along_axis(self.A[j][o],ind.reshape(self.n_timesteps,1,1),axis=1)@grad_c_k).squeeze(-2)# shape: n_timsteps, n_links                    
                    
                    self.cons[(j+self.n_links*o)*self.n_timesteps:(j+self.n_links*o+1)*self.n_timesteps] = cons
                    self.jac[(j+self.n_links*o)*self.n_timesteps:(j+self.n_links*o+1)*self.n_timesteps] = jac
           
            self.x_prev = np.copy(x)   





class NlpSetup3D(NlpSetup):
    def __init__(self,A,b,FO_link,qpos,qvel,qgoal,n_timesteps,n_links,n_obs, n_pos_lim, actual_pos_lim, vel_lim, lim_flag, dimension, g_ka):
        self.A = A 
        self.b = b
        self.qpos = qpos
        self.qvel = qvel 
        self.qgoal = qgoal 
        self.x_prev = np.zeros(n_links)*np.nan
        self.n_timesteps = n_timesteps
        self.n_links = n_links 
        self.n_obs = n_obs 
        self.n_obs_cons = self.n_timesteps*self.n_obs
        self.M_obs = self.n_links*self.n_obs_cons
        self.M = self.M_obs+2*self.n_links+6*n_pos_lim
        self.dimension = dimension
        self.g_ka = g_ka
        self.parse_FO_link(FO_link)

        self.actual_pos_lim = actual_pos_lim
        self.vel_lim = vel_lim
        self.lim_flag = lim_flag

    def objective(self,x):
        qplan = self.qpos + self.qvel*T_PLAN + 0.5*self.g_ka*x*T_PLAN**2
        return np.sum(wrap_cont_joint_to_pi(qplan-self.qgoal,self.lim_flag)**2)

    def gradient(self,x):
        qplan = self.qpos + self.qvel*T_PLAN + 0.5*self.g_ka*x*T_PLAN**2
        return self.g_ka* T_PLAN**2*wrap_cont_joint_to_pi(qplan - self.qgoal,self.lim_flag)

    def compute_constraints_jacobian(self,x):
        if (self.x_prev!=x).any(): 
            self.cons = np.zeros((self.M))
            self.jac = np.zeros((self.M, self.n_links))
            
            # position and velocity constraints
            t_peak_optimum = -self.qvel/(self.g_ka*x) # time to optimum of first half traj.
            qpos_peak_optimum = np.nan_to_num((t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(self.qpos+self.qvel*t_peak_optimum+0.5*(self.g_ka*x)*t_peak_optimum**2), 0)
            grad_qpos_peak_optimum = np.diag(np.nan_to_num((t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(0.5*self.g_ka*t_peak_optimum**2),0))
            qpos_peak = self.qpos + self.qvel * T_PLAN + 0.5 * (self.g_ka * x) * T_PLAN**2
            grad_qpos_peak = 0.5 * self.g_ka * T_PLAN**2 * np.eye(self.n_links)
            qvel_peak = self.qvel + self.g_ka * x * T_PLAN
            grad_qvel_peak = self.g_ka * T_PLAN * np.eye(self.n_links)

            bracking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
            qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*bracking_accel*(T_FULL-T_PLAN)**2
            # can be also, qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL+T_PLAN) + 0.5 * (self.g_ka * x) * T_PLAN * T_FULL
            grad_qpos_brake = 0.5 * self.g_ka * T_PLAN * T_FULL * np.eye(self.n_links)

            qpos_possible_max_min = np.vstack((qpos_peak_optimum,qpos_peak,qpos_brake))[:,self.lim_flag] 
            qpos_ub = (qpos_possible_max_min - self.actual_pos_lim[:,0]).flatten()
            qpos_lb = (self.actual_pos_lim[:,1] - qpos_possible_max_min).flatten()
            
            grad_qpos_ub = np.vstack((grad_qpos_peak_optimum[self.lim_flag],grad_qpos_peak[self.lim_flag],grad_qpos_brake[self.lim_flag]))
            grad_qpos_lb = - grad_qpos_ub

            self.cons[self.M_obs:] = np.hstack((qvel_peak-self.vel_lim, -self.vel_lim-qvel_peak,qpos_ub,qpos_lb))
            self.jac[self.M_obs:] = np.vstack((grad_qvel_peak, -grad_qvel_peak, grad_qpos_ub, grad_qpos_lb))                    

            if self.n_obs > 0:
            #if True:
                for j in range(self.n_links):
                    c_k = np.expand_dims(self.FO_center_slice_all_dep(x,j),-1)
                    grad_c_k = self.FO_grad_center_slice_all_dep(x,j)
                    h_obs = (self.A[j]@c_k).squeeze(-1) - self.b[j]
                    
                    ind = np.argmax(np.nan_to_num(h_obs,-np.inf),-1) 
                    cons = - np.take_along_axis(h_obs,ind.reshape(self.n_obs,self.n_timesteps,1),axis=-1).squeeze(-1) # shape: n_obs, n_timsteps, SAFE if <=-1e-6 
                    jac = - (np.take_along_axis(self.A[j],ind.reshape(self.n_obs,self.n_timesteps,1,1),axis=-2)@grad_c_k).squeeze(-2)# shape: n_obs, n_timsteps, n_links                    

                    self.cons[j*self.n_obs_cons:(j+1)*self.n_obs_cons] = cons.reshape(self.n_obs_cons)
                    self.jac[j*self.n_obs_cons:(j+1)*self.n_obs_cons]  = jac.reshape(self.n_obs_cons,self.n_links)
        
            self.x_prev = np.copy(x)



class NlpSetupLocked3D(NlpSetup3D):
    def __init__(self,A,b,FO_link,qpos,qvel,qgoal,n_timesteps,n_links,dof,n_obs, n_pos_lim, actual_pos_lim, vel_lim, lim_flag, dimension, g_ka):
        
        super().__init__(
            A,
            b,
            FO_link,
            qpos,
            qvel,
            qgoal,
            n_timesteps,
            n_links,
            n_obs,
            n_pos_lim,
            actual_pos_lim,
            vel_lim,
            lim_flag,
            dimension,
            g_ka)
        self.dof = dof
        self.x_prev = np.zeros(self.dof)*np.nan
        self.M_obs = self.n_links*self.n_obs_cons
        self.M = self.M_obs+2*self.dof+6*n_pos_lim
        

    def compute_constraints_jacobian(self,x):

        if (self.x_prev!=x).any(): 
        
            self.cons = np.zeros((self.M))
            self.jac = np.zeros((self.M, self.dof))
            
            # position and velocity constraints
            t_peak_optimum = -self.qvel/(self.g_ka*x) # time to optimum of first half traj.
            qpos_peak_optimum = np.nan_to_num((t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(self.qpos+self.qvel*t_peak_optimum+0.5*(self.g_ka*x)*t_peak_optimum**2), 0)
            grad_qpos_peak_optimum = np.diag(np.nan_to_num((t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(0.5*self.g_ka*t_peak_optimum**2),0))
            qpos_peak = self.qpos + self.qvel * T_PLAN + 0.5 * (self.g_ka * x) * T_PLAN**2
            grad_qpos_peak = 0.5 * self.g_ka * T_PLAN**2 * np.eye(self.dof)
            qvel_peak = self.qvel + self.g_ka * x * T_PLAN
            grad_qvel_peak = self.g_ka * T_PLAN * np.eye(self.dof)

            bracking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
            qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*bracking_accel*(T_FULL-T_PLAN)**2
            # can be also, qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL+T_PLAN) + 0.5 * (self.g_ka * x) * T_PLAN * T_FULL
            grad_qpos_brake = 0.5 * self.g_ka * T_PLAN * T_FULL * np.eye(self.dof)

            qpos_possible_max_min = np.vstack((qpos_peak_optimum,qpos_peak,qpos_brake))[:,self.lim_flag] 
            qpos_ub = (qpos_possible_max_min - self.actual_pos_lim[:,0]).flatten()
            qpos_lb = (self.actual_pos_lim[:,1] - qpos_possible_max_min).flatten()
            
            grad_qpos_ub = np.vstack((grad_qpos_peak_optimum[self.lim_flag],grad_qpos_peak[self.lim_flag],grad_qpos_brake[self.lim_flag]))
            grad_qpos_lb = - grad_qpos_ub

            self.cons[self.M_obs:] = np.hstack((qvel_peak-self.vel_lim, -self.vel_lim-qvel_peak,qpos_ub,qpos_lb))
            self.jac[self.M_obs:] = np.vstack((grad_qvel_peak, -grad_qvel_peak, grad_qpos_ub, grad_qpos_lb))                    

            if self.n_obs > 0:
                for j in range(self.n_links):
                    c_k = np.expand_dims(self.FO_center_slice_all_dep(x,j),-1)
                    grad_c_k = self.FO_grad_center_slice_all_dep(x,j)
                    h_obs = (self.A[j]@c_k).squeeze(-1) - self.b[j]
                    
                    ind = np.argmax(np.nan_to_num(h_obs,-np.inf),-1) 
                    cons = - np.take_along_axis(h_obs,ind.reshape(self.n_obs,self.n_timesteps,1),axis=-1).squeeze(-1) # shape: n_obs, n_timsteps, SAFE if <=-1e-6 
                    jac = - (np.take_along_axis(self.A[j],ind.reshape(self.n_obs,self.n_timesteps,1,1),axis=-2)@grad_c_k).squeeze(-2)# shape: n_obs, n_timsteps, n_links                    

                    self.cons[j*self.n_obs_cons:(j+1)*self.n_obs_cons] = cons.reshape(self.n_obs_cons)
                    self.jac[j*self.n_obs_cons:(j+1)*self.n_obs_cons]  = jac.reshape(self.n_obs_cons,self.dof)
        
            self.x_prev = np.copy(x)