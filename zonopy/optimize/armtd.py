import torch
import zonopy as zp
from zonopy.kinematics.FO import forward_occupancy

from ._constraints import collision_constraint
class ARMTD_planner():
    def __init__(self,params,zono_order,max_combs):
        self.params=params 
        
        
        self.zono_order = zono_order
        self.max_combs = max_combs


    def generate_combinations_upto(self):
        combs = [torch.tensor([0])]
        for i in range(1,self.max_combs):
            combs.append(torch.combinations(torch.arange(i+1),2))

    def create_FRS(self,qpos,qvel):
        _, R_trig = zp.load_batch_JRS_trig_ic(qpos,qvel)
        FO_link,_, _ = forward_occupancy(R_trig,self.params)
        self.qpos = qpos 
        self.qvel = qvel

    def trajopt(self,qgoal,ka_0):
        class nlp_setup():
            def objective(p,ka):
                obj = qgoal - (self.qpos)
                return obj**2
        
        nlp = cyipopt.Problem(
        n=len(x0),
        m=len(cl),
        problem_obj=nlp_obj(),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
        )
        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('tol', 1e-7)
        x, info = nlp.solve(x0)


def ARMTD(qpos,qvel,qgoal, obstacles, params, ka_i):
    _, R_trig = zp.load_batch_JRS_trig_ic(qpos,qvel)
    FO_link,_, _ = forward_occupancy(R_trig,params)
    N_joints = len(qpos)
    N_obs = len(obstacles)
    A,b = [],[]
    for j in range(N_joints):
        for o in range(N_obs):
            A.append()


    class nlp_setup():
        def objective(self,ka):
            obj = qgoal - (qpos+ka*qvel*0.5+1/2*1a)
            return obj**2

        def constraint(self, ka):
            return ka 




    nlp = cyipopt.Problem(
    n=len(x0),
    m=len(cl),
    problem_obj=nlp_obj(),
    lb=lb,
    ub=ub,
    cl=cl,
    cu=cu,
    )
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-7)
    x, info = nlp.solve(x0)
