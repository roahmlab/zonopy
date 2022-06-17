import torch
import numpy as np
from zonopy.kinematics.FO import forward_occupancy
from zonopy.joint_reachable_set.jrs_trig.process_jrs_trig import process_batch_JRS_trig_ic
from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import preload_batch_JRS_trig
from zonopy.conSet.zonotope.batch_zono import batchZonotope
import zonopy as zp
import cyipopt

import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1.0

# batch

def gen_grad_RTS_2D_Layer(link_zonos,joint_axes,n_links,n_obs,params):

    jrs_tensor = preload_batch_JRS_trig()
    dimension = 2
    n_timesteps = 100
    ka_0 = torch.zeros(n_links)
    PI_vel = torch.tensor(torch.pi-1e-6)
    g_ka = torch.pi/24

    class grad_RTS_2D_Layer(torch.autograd.Function):
        @staticmethod


        def forward(ctx,lambd,observation):
            # observation = [ qpos | qvel | qgoal | obs_pos1,...,obs_posO | obs_size1,...,obs_sizeO ]
            zp.reset()
            ctx.lambd_shape, ctx.obs_shape = lambd.shape, observation.shape
            ctx.lambd =lambd.clone().reshape(-1,n_links).to(dtype=torch.get_default_dtype())             
            #observation = observation.reshape(-1,observation.shape[-1]).to(dtype=torch.get_default_dtype())
            observation = observation.to(dtype=torch.get_default_dtype())
            ka = g_ka*ctx.lambd
            
            n_batches = observation.shape[0]
            ctx.qpos = observation[:,:n_links]
            ctx.qvel = observation[:,n_links:2*n_links]
            obstacle_pos = observation[:,-4*n_obs:-2*n_obs]
            obstacle_size = observation[:,-2*n_obs:]
            qgoal = ctx.qpos + ctx.qvel*T_PLAN + 0.5*ka*T_PLAN**2

            #g_ka = torch.maximum(PI/24,abs(qvel/3))

            _, R_trig = process_batch_JRS_trig_ic(jrs_tensor,ctx.qpos,ctx.qvel,joint_axes)
            FO_link,_,_ = forward_occupancy(R_trig,link_zonos,params)
            
            ctx.As = [[] for _ in range(n_links)]
            bs = [[] for _ in range(n_links)]

            lambda_to_slc = ctx.lambd.reshape(n_batches,1,dimension).repeat(1,n_timesteps,1)
            
            #unsafe_flag = torch.zeros(n_batches) 
            unsafe_flag = (abs(ctx.qvel+ctx.lambd*g_ka*T_PLAN)>PI_vel).any(-1)#NOTE: this might not work on gpu, velocity lim check
            for j in range(n_links):
                FO_link[j] = FO_link[j].project([0,1]) 
                c_k = FO_link[j].center_slice_all_dep(lambda_to_slc).unsqueeze(-1) # FOR, safety check
                for o in range(n_obs):
                    obs_Z = torch.cat((obstacle_pos[:,2*o:2*(o+1)].unsqueeze(-2),torch.diag_embed(obstacle_size[:,2*o:2*(o+1)])),-2).unsqueeze(-3).repeat(1,n_timesteps,1,1)
                    A_temp, b_temp = batchZonotope(torch.cat((obs_Z,FO_link[j].Grest),-2)).polytope() # A: n_timesteps,*,dimension                     
                    ctx.As[j].append(A_temp)
                    bs[j].append(b_temp)
                    unsafe_flag += (torch.max((A_temp@c_k).squeeze(-1)-b_temp,-1)[0]<1e-6).any(-1)  #NOTE: this might not work on gpu FOR, safety check

            #unsafe_flag = torch.ones(n_batches,dtype=bool) # NOTE: activate rts all ways

            M_obs = n_timesteps*n_links*n_obs
            M = M_obs+2*n_links
            ctx.flags = -torch.ones(n_batches,dtype=int) # -1: direct pass, 0: safe plan from armtd pass, 1: fail-safe plan from armtd pass
            ctx.nlp_obj = [None for _ in range(n_batches)]
            ctx.nlp_info = [None for _ in range(n_batches)]
            for i in unsafe_flag.nonzero().reshape(-1):
                class nlp_setup():
                    x_prev = np.zeros(n_links)*np.nan
                    def objective(nlp,x):
                        qplan = ctx.qpos[i] + ctx.qvel[i]*T_PLAN + 0.5*g_ka*x*T_PLAN**2
                        return torch.sum(wrap_to_pi(qplan-qgoal[i])**2)

                    def gradient(nlp,x):
                        qplan = ctx.qpos[i] + ctx.qvel[i]*T_PLAN + 0.5*g_ka*x*T_PLAN**2
                        return (g_ka*T_PLAN**2*wrap_to_pi(qplan-qgoal[i])).numpy()

                    def constraints(nlp,x): 
                        ka = torch.tensor(x,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(n_timesteps,1)
                        if (nlp.x_prev!=x).any():                
                            nlp.possible_obs_cons = []# NOTE
                            nlp.obs_cons_max_ind = torch.zeros(n_links,n_obs,n_timesteps,dtype=int)# NOTE
                            Cons = torch.zeros(M)   
                            Jac = torch.zeros(M,n_links)
                            # velocity constraints
                            q_peak = ctx.qvel[i]+g_ka*x*T_PLAN
                            grad_q_peak = g_ka*T_PLAN*torch.eye(n_links)
                            Cons[-2*n_links:] = torch.hstack((q_peak-torch.pi,-torch.pi-q_peak))
                            Jac[-2*n_links:] = torch.vstack((grad_q_peak,-grad_q_peak))
                            # velocity constraints 
                            for j in range(n_links):
                                c_k = FO_link[j][i].center_slice_all_dep(ka)
                                grad_c_k = FO_link[j][i].grad_center_slice_all_dep(ka)
                                for o in range(n_obs):
                                    pos_obs_cons = (ctx.As[j][o][i]@c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o][i]
                                    nlp.possible_obs_cons.extend(list(pos_obs_cons))
                                    cons, nlp.obs_cons_max_ind[j,o]= torch.max(pos_obs_cons,-1) # shape: n_timsteps, SAFE if >=1e-6
                                    jac = (ctx.As[j][o][i].gather(-2,nlp.obs_cons_max_ind[j,o].reshape(n_timesteps,1,1).repeat(1,1,dimension))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                                    Cons[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = - cons
                                    Jac[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = - jac
                            nlp.cons = Cons.numpy()
                            nlp.jac = Jac.numpy()
                            nlp.x_prev = np.copy(x)                   
                        return nlp.cons

                    def jacobian(nlp,x): 
                        ka = torch.tensor(x,dtype=torch.get_default_dtype()).unsqueeze(0).repeat(n_timesteps,1)
                        if (nlp.x_prev!=x).any():                
                            nlp.possible_obs_cons = []# NOTE
                            nlp.obs_cons_max_ind = torch.zeros(n_links,n_obs,n_timesteps,dtype=int)# NOTE
                            Cons = torch.zeros(M)   
                            Jac = torch.zeros(M,n_links)
                            # velocity constraints 
                            q_peak = ctx.qvel[i]+g_ka*x*T_PLAN
                            grad_q_peak = g_ka*T_PLAN*torch.eye(n_links)
                            Cons[-2*n_links:] = torch.hstack((q_peak-torch.pi,-torch.pi-q_peak))
                            Jac[-2*n_links:] = torch.vstack((grad_q_peak,-grad_q_peak))
                            # velocity constraints 
                            for j in range(n_links):
                                c_k = FO_link[j][i].center_slice_all_dep(ka)
                                grad_c_k = FO_link[j][i].grad_center_slice_all_dep(ka)
                                for o in range(n_obs):
                                    pos_obs_cons = (ctx.As[j][o][i]@c_k.unsqueeze(-1)).squeeze(-1) - bs[j][o][i]
                                    nlp.possible_obs_cons.extend(list(pos_obs_cons))
                                    cons, nlp.obs_cons_max_ind[j,o]= torch.max(pos_obs_cons,-1) # shape: n_timsteps, SAFE if >=1e-6
                                    jac = (ctx.As[j][o][i].gather(-2,nlp.obs_cons_max_ind[j,o].reshape(n_timesteps,1,1).repeat(1,1,dimension))@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                                    Cons[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = - cons
                                    Jac[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = - jac
                            nlp.cons = Cons.numpy()
                            nlp.jac = Jac.numpy()
                            nlp.x_prev = np.copy(x)            
                        return nlp.jac
                    
                    def intermediate(nlp, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                            d_norm, regularization_size, alpha_du, alpha_pr,
                            ls_trials):
                        pass
                

                ctx.nlp_obj[i] = nlp_setup()
                NLP = cyipopt.problem(
                n = n_links,
                m = M,
                problem_obj=ctx.nlp_obj[i],
                lb = [-1]*n_links,
                ub = [1]*n_links,
                cl = [-1e20]*M_obs+[-1e20]*n_links+[-1e20]*n_links,
                cu = [-1e-6]*M_obs+[-1e-6]*n_links+[-1e-6]*n_links,
                )
                NLP.addOption('sb', 'yes')
                NLP.addOption('print_level', 0)
                
                k_opt, ctx.nlp_info[i] = NLP.solve(ka_0)


                # NOTE: for training, dont care about fail-safe
                if ctx.nlp_info[i]['status'] == 0:
                    ctx.lambd[i] = torch.tensor(k_opt,dtype = torch.get_default_dtype())
                    ctx.flags[i] = 0
                else:
                    ctx.flags[i] = 1
                      
            return ctx.lambd, FO_link, ctx.flags

        @staticmethod
        def backward(ctx,*grad_ouput):
            direction = grad_ouput[0]
            grad_input = torch.zeros_like(direction)

            M_obs = n_timesteps*n_links*n_obs
            for i, flag in enumerate(ctx.flags):
                if flag == -1: # direct pass
                    grad_input[i] = direction[i]
                elif flag == 0: # rts success path. solve QP

                    k_opt = ctx.lambd[i].numpy()
                    # COMPUTE GRADIENT
                    tol = 1e-6
                    # compute jacobian of each smooth constraint which will be constraints for QP
                    jac = ctx.nlp_obj[i].jacobian(k_opt)
                    cons = ctx.nlp_obj[i].cons
                    A_AT = []
                    size_As = torch.zeros(n_links,n_obs,dtype=int)
                    for j in range(n_links):
                        for o in range(n_obs):
                            A = ctx.As[j][o][i]
                            size_As[j,o] = A.shape[-2]
                            a_at = 2*(A.gather(-2,ctx.nlp_obj[i].obs_cons_max_ind[j,o].reshape(n_timesteps,1,1).repeat(1,1,dimension))@A.transpose(-2,-1)).squeeze(-2)
                            A_AT.extend(list(a_at))
                    size_As = size_As.unsqueeze(-1).repeat(1,1,n_timesteps).flatten()
                    size_As = torch.hstack((torch.zeros(1,dtype=int),size_As)).cumsum(0)
                    
                    num_smooth_var = int(size_As[-1]) # full dimension of lambda
                    num_a_var = n_links # number of decision var. in armtd
                    num_b_var = num_a_var + num_smooth_var # number of decition var. in B-armtd
                    
                    qp_cons1 = np.hstack((jac[:M_obs],- torch.block_diag(*ctx.nlp_obj[i].possible_obs_cons).numpy().reshape(M_obs,-1))) # [A*c(k)-b].T*lambda
                    qp_cons2 = np.hstack((np.zeros((M_obs,n_links)),torch.block_diag(*A_AT).numpy())) # ||A.T*lambda||-1
                    EYE = np.eye(num_b_var)
                    #qp_cons3 = sp.csr_matrix(([1.]*size_As[-1],(range(qp_cons1.shape[-1]-n_links),range(n_links,qp_cons1.shape[-1]))))
                    qp_cons3 = EYE[num_a_var:] # lambda
                    qp_cons4 = -EYE[:num_a_var] # lb
                    qp_cons5 = EYE[:num_a_var] # ub
                    qp_cons6 =  np.hstack((jac[-2*n_links:],np.zeros((2*n_links,num_smooth_var))))
                    qp_cons = np.vstack((qp_cons1,qp_cons2,qp_cons3,qp_cons4,qp_cons5,qp_cons6))

                    # compute duals for smooth constraints                
                    mult_smooth_cons1 = ctx.nlp_info[i]['mult_g'][:M_obs]*(ctx.nlp_info[i]['mult_g'][:M_obs]>tol)
                    mult_smooth_cons2 = np.zeros(M_obs)
                    mult_smooth_cons3 = np.zeros(num_smooth_var)
                    

                    for idx in range(M_obs):
                        mult_smooth_cons3[size_As[idx]:size_As[idx+1]] = -mult_smooth_cons1[idx]*ctx.nlp_obj[i].possible_obs_cons[idx] # NOTE
                    mult_smooth_cons4 = ctx.nlp_info[i]['mult_x_L']*(ctx.nlp_info[i]['mult_x_L']>tol)
                    mult_smooth_cons5 = ctx.nlp_info[i]['mult_x_U']*(ctx.nlp_info[i]['mult_x_U']>tol)
                    mult_smooth_cons6 = ctx.nlp_info[i]['mult_g'][-2*n_links:]*(ctx.nlp_info[i]['mult_g'][-2*n_links:]>tol)

                    mult_smooth = np.hstack((mult_smooth_cons1,mult_smooth_cons2,mult_smooth_cons3,mult_smooth_cons4,mult_smooth_cons5,mult_smooth_cons6))
                    
                    # compute smooth constraints     
                    smoother = np.zeros(num_smooth_var) # NOTE: we might wanna assign smoother value for inactive or weakly active as 1/2 instead of 1.
                    obs_cons_max_inds = size_As[:-1]+ctx.nlp_obj[i].obs_cons_max_ind.flatten()
                    smoother[obs_cons_max_inds] = 1
                    
                    smooth_cons1 = cons[:M_obs]*(cons[:M_obs]<-1e-6-tol)
                    smooth_cons2 = np.zeros(M_obs)
                    ''' 
                    # This will result in all zero if all the nonzero element of smoother is 1
                    for j in range(n_links):
                        for o in range(n_obs):
                            A_smoother = As[j][o][i].gather(-2,ctx.nlp_obj.obs_cons_max_ind[j,o].reshape(n_timesteps,1,1).repeat(1,1,dimension)).squeeze(-2)
                            smooth_cons2[(j+n_links*o)*n_timesteps:(j+n_links*o+1)*n_timesteps] = torch.linalg.norm(A_smoother,dim=-1)**2-1        
                    '''
                    smooth_cons3 = -smoother
                    smooth_cons4 = (- 1 - k_opt) * (- 1 - k_opt <-1e-6-tol)
                    smooth_cons5 = (k_opt - 1) * (k_opt - 1 <-1e-6-tol)
                    smooth_cons6 = cons[-2*n_links:]*(cons[-2*n_links:]<-1e-6-tol)
                    smooth_cons = np.hstack((smooth_cons1,smooth_cons2,smooth_cons3,smooth_cons4,smooth_cons5,smooth_cons6))
                    
                    # compute cost for QP: no alph, constant g_k, so we can simplify cost fun.
                    H = 0.5*sp.csr_matrix(([1.]*num_a_var,(range(num_a_var),range(num_a_var))),shape=(num_b_var,num_b_var))
                    f_d = sp.csr_matrix((-direction[i].numpy(),([0.]*num_a_var,range(num_a_var))),shape=(1,num_b_var))

                    strongly_active = (mult_smooth > tol) * (smooth_cons >= -1e-6-tol)
                    weakly_active = (mult_smooth <= tol) * (smooth_cons >= -1e-6-tol)
                    inactive = (smooth_cons < -1e-6-tol)

                    # QP API
                    qp = gp.Model("back_prop")
                    qp.Params.LogToConsole = 0 # turn off printing
                    z = qp.addMVar(shape=num_b_var, name="z",vtype=GRB.CONTINUOUS,ub=np.inf, lb=-np.inf)
                    qp.setObjective(z@H@z+f_d@z, GRB.MINIMIZE)
                    qp_eq_cons = sp.csr_matrix(qp_cons[strongly_active])
                    rhs_eq = np.zeros(strongly_active.sum())
                    qp_ineq_cons = sp.csr_matrix(qp_cons[weakly_active])
                    rhs_ineq = -0*np.ones(weakly_active.sum())
                    qp.addConstr( qp_eq_cons @ z == rhs_eq, name="eq")
                    qp.addConstr(qp_ineq_cons @ z <= rhs_ineq, name="ineq")
                    qp.optimize()
                    grad_input[i] = torch.tensor(z.X[:num_a_var],dtype = torch.get_default_dtype())
                    
                # NOTE: for fail-safe, keep into zeros 

            return (grad_input.reshape(ctx.lambd_shape),torch.zeros(ctx.obs_shape))
    return grad_RTS_2D_Layer.apply

if __name__ == '__main__':
    from zonopy.environments.arm_2d import Arm_2D
    import time
    n_links = 2
    env = Arm_2D(n_links=n_links,n_obs=1)
    observation = env.set_initial(qpos = torch.tensor([0.1*torch.pi,0.1*torch.pi]),qvel= torch.zeros(n_links), qgoal = torch.tensor([-0.5*torch.pi,-0.8*torch.pi]),obs_pos=[torch.tensor([-1,-0.9])])
    
    t_armtd = 0
    params = {'n_joints':env.n_links, 'P':env.P0, 'R':env.R0}
    joint_axes = [j for j in env.joint_axes]
    RTS = gen_grad_RTS_2D_Layer(env.link_zonos,joint_axes,env.n_links,env.n_obs,params)

    n_steps = 30
    for _ in range(n_steps):
        ts = time.time()
        observ_temp = torch.hstack([observation[key].flatten() for key in observation.keys() ])
        #k = 2*(env.qgoal - env.qpos - env.qvel*T_PLAN)/(T_PLAN**2)
        lam = torch.tensor([0.8,0.8])
        bias1 = torch.full((3,1),0.0,requires_grad=True )
        lam, FO_link, flag = RTS(torch.vstack((lam,lam,lam))+bias1,torch.vstack((observ_temp,observ_temp,observ_temp))+bias1) 
        lam.sum().backward()

        import pdb;pdb.set_trace()
        #ka, FO_link, flag = RTS(k,observ_temp)
        print(f'action: {lam}')
        print(f'flag: {flag}')
            
        t_elasped = time.time()-ts
        print(f'Time elasped for ARMTD-2d:{t_elasped}')
        t_armtd += t_elasped
        #print(ka[0])
        observation, reward, done, info = env.step(lam[0]*torch.pi/24,flag[0])
        
        FO_link = [fo[0] for fo in FO_link]
        env.render(FO_link)
        '''
        if done:
            import pdb;pdb.set_trace()
            break
        '''


    print(f'Total time elasped for ARMTD-2d with {n_steps} steps: {t_armtd}')
    import pdb;pdb.set_trace()

