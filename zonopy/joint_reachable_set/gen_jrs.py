import torch
from zonopy.transformations.rotation import gen_rotatotope_from_jrs
from zonopy.joint_reachable_set.utils import remove_dependence_and_compress
import zonopy as zp
PI = torch.tensor(torch.pi)


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
                 k_r=None):
        self.robot = robot
        self.traj = traj_class

        self.num_q = len(robot.actuated_joints)
        self.joint_axis = [joint.axis for joint in robot.actuated_joints]
        self.joint_axis = np.array(self.joint_axis)

        self.param_range = np.ones(self.num_q) * param_range
        self.param_range = np.vstack(
            [param_center - self.param_range, param_center + self.param_range]
        )

        self.tplan = tplan
        self.tfinal = tfinal
        self.tdiscretization = tdiscretization
        num_t = int(tfinal/tdiscretization)

        self.id_map = OrderedDict()

        self.ultimate = (ultimate_bound, k_r) if ultimate_bound is not None and k_r is not None else None

        # Create base PZ's for parameters
        self.base_params = np.empty(self.num_q, dtype=object)
        for i in range(self.num_q):
            id = len(self.id_map)
            self.id_map[f'k{i}'] = id
            self.base_params[i] = zp.polyZonotope(
                [[0],[1]],
                1, id=id
            )

        # Create base PZ's for time
        self.times = np.empty(num_t, dtype=object)
        for i in range(num_t):
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
    
    def gen_JRS(self, q_in, qd_in, qdd_in=None, taylor_degree=1, make_gens_independent=True):
        q_in = q_in.squeeze()
        qd_in = qd_in.squeeze()
        qdd_in = qdd_in.squeeze() if qdd_in is not None else None

        # create the trajectory generator
        gen = self.traj(
            q_in, qd_in, qdd_in,
            self.base_params, self.param_range,
            self.tplan, self.tfinal
        )

        # Get the reference trajectory
        print('Creating Reference Trajectory')
        q_ref, qd_ref, qdd_ref = gen.getReference(self.times)

        # Vectorize over q
        rot_from_q = np.vectorize(JrsGenerator._get_pz_rotations_from_q, excluded=[1,'taylor_deg'])

        print('Creating Reference Rotatotopes')
        R_ref = np.empty_like(q_ref)
        for i in range(self.num_q):
            R_ref[:,i] = rot_from_q(q_ref[:,i], self.joint_axis[i], taylor_deg=taylor_degree)

        # Add tracking error if provided
        if self.ultimate is not None:
            print('Adding tracking error to make tracked trajectory')
            q = q_ref + self.error_pos
            qd = qd_ref + self.error_vel
            qd_aux = qd_ref + self.ultimate[1] * self.error_pos
            qdd_aux = qdd_ref + self.ultimate[1] * self.error_vel
            print('Creating Output Rotatotopes')
            R = np.empty_like(q)
            for i in range(self.num_q):
                R[:,i] = rot_from_q(q[:,i], self.joint_axis[i], taylor_deg=taylor_degree)
        else:
            q, qd = q_ref, qd_ref
            qd_aux, qdd_aux = qd_ref, qdd_ref
            R = R_ref

        # Return as dict
        return {
            'q_ref': q_ref,
            'qd_ref': qd_ref,
            'qdd_ref': qdd_ref,
            'R_ref': R_ref,
            'q': q,
            'qd': qd,
            'qd_aux': qd_aux,
            'qdd_aux': qdd_aux,
            'R': R
        }
    
    @staticmethod
    def _get_pz_rotations_from_q(q, rotation_axis, taylor_deg=6):
        cos_q = JrsGenerator._cos(q, taylor_deg)
        sin_q = JrsGenerator._sin(q, taylor_deg)
        cos_sin_q = cos_q.exactCartProd(sin_q)
        # Rotation Matrix
        e = rotation_axis/np.linalg.norm(rotation_axis)
        U = np.array([[0, -e[2], e[1]],[e[2], 0, -e[0]],[-e[1], e[0], 0]])

        # Create rotation matrices from cos and sin dimensions
        cq = cos_sin_q.c[0]
        sq = cos_sin_q.c[1]
        C = torch.eye(3) + sq*U + (1-cq)*U@U
        C = C.unsqueeze(0)

        tmp_Grest = None
        if cos_sin_q.Grest.shape[0] > 0:
            cq = cos_sin_q.Grest[:,0].reshape([-1,1,1])
            sq = cos_sin_q.Grest[:,1].reshape([-1,1,1])
            tmp_Grest = sq*U + -cq*U@U

        tmp_G = None
        if cos_sin_q.G.shape[0] > 0:
            cq = cos_sin_q.G[:,0].reshape([-1,1,1])
            sq = cos_sin_q.G[:,1].reshape([-1,1,1])
            tmp_G = sq*U + -cq*U@U
        
        Z = torch.concat((C, tmp_G, tmp_Grest), dim=0)
        # Z_t = Z.transpose(1,2)
        out = zp.matPolyZonotope(Z, cos_sin_q.n_dep_gens, cos_sin_q.expMat, cos_sin_q.id, compress=0)
        # out_t = zp.matPolyZonotope(Z_t, cos_sin_q.n_dep_gens, cos_sin_q.expMat, cos_sin_q.id, compress=0)
        # return (out, out_t)
        return out

    # Put this here for now, but eventually find a better place to put this
    @staticmethod
    def _cos(pz, order=6):
        # Make sure we're only using 1D pz's
        assert pz.dimension == 1, "Operation only valid for a 1D PZ"
        pz_c = torch.cos(pz.c)

        out = pz_c

        cs_cf = torch.cos(pz.c)
        sn_cf = torch.sin(pz.c)

        def sgn_cs(n):
            if n%4 == 0 or n%4 == 3:
                return 1
            else:
                return -1
            
        factor = 1
        T_factor = 1
        pz_neighbor = pz - pz.c

        for i in range(order):
            factor = factor * (i + 1)
            T_factor = T_factor * pz_neighbor
            if i % 2:
                out = out + (sgn_cs(i+1) * cs_cf / factor) * T_factor
            else:
                out = out + (sgn_cs(i+1) * sn_cf / factor) * T_factor

        # add lagrange remainder interval to Grest
        rem = pz_neighbor.to_interval()
        rem_pow = (T_factor * pz_neighbor).to_interval()

        # NOTE zp cos and sin working for interval but not pz
        if order % 2 == 0:
            J0 = zp.sin(pz.c + zp.interval([0], [1]) * rem)
        else:
            J0 = zp.cos(pz.c + zp.interval([0], [1]) * rem)
        
        if order % 4 == 0 or order % 4 == 1:
            J = -J0
        else:
            J = J0

        remainder = 1. / (factor * (order + 1)) * rem_pow * J

        # Assumes a 1D pz
        c = out.c + remainder.center()
        G = out.G
        Grest = torch.sum(out.Grest) + remainder.rad()
        Z = torch.vstack([c, G, Grest])
        out = zp.polyZonotope(Z, out.n_dep_gens, out.expMat, out.id)
        return out

    # Put this here for now, but eventually find a better place to put this
    @staticmethod
    def _sin(pz, order=6):
        # Make sure we're only using 1D pz's
        assert pz.dimension == 1, "Operation only valid for a 1D PZ"
        pz_c = torch.sin(pz.c)

        out = pz_c

        cs_cf = torch.cos(pz.c)
        sn_cf = torch.sin(pz.c)

        def sgn_cs(n):
            if n%4 == 0 or n%4 == 1:
                return 1
            else:
                return -1
            
        factor = 1
        T_factor = 1
        pz_neighbor = pz - pz.c

        for i in range(order):
            factor = factor * (i + 1)
            T_factor = T_factor * pz_neighbor
            if i % 2 == 0:
                out = out + (sgn_cs(i+1) * cs_cf / factor) * T_factor
            else:
                out = out + (sgn_cs(i+1) * sn_cf / factor) * T_factor

        # add lagrange remainder interval to Grest
        rem = pz_neighbor.to_interval()
        rem_pow = (T_factor * pz_neighbor).to_interval()

        # TODO make interval sine and cosine
        if order % 2 == 1:
            J0 = zp.sin(pz.c + zp.interval([0], [1]) * rem)
        else:
            J0 = zp.cos(pz.c + zp.interval([0], [1]) * rem)
        
        if order % 4 == 1 or order % 4 == 2:
            J = -J0
        else:
            J = J0

        remainder = 1. / (factor * (order + 1)) * rem_pow * J

        # Assumes a 1D pz
        c = out.c + remainder.center()
        G = out.G
        Grest = torch.sum(out.Grest) + remainder.rad()
        Z = torch.vstack([c, G, Grest])
        out = zp.polyZonotope(Z, out.n_dep_gens, out.expMat, out.id)
        return out




def gen_JRS(q,dq,joint_axes=None,taylor_degree=1,make_gens_independent=True):
    n_q = len(q)
    if joint_axes is None:
        joint_axes = [torch.tensor([0.0,0.0,1.0]) for _ in range(n_q)]
    
    traj_type = 'orig'
    T_full,T_plan, dt = 1, 0.5 ,0.01
    n_t = int(T_full/dt)
    n_t_p = int(T_plan/T_full*n_t)

    c_k = torch.zeros(n_q)
    g_k = torch.min(torch.max(PI/24,abs(dq/3)),PI/3)
    T, K = [],[]
    Q, R= [[[] for _ in range(n_t)]for _ in range(2)]
    for _ in range(n_q):
        K.append(zp.polyZonotope([0],[[1]],prop='k'))
    for t in range(n_t):
        T.append(zp.polyZonotope([dt*t+dt/2],[[dt/2]]))

    Qd_plan, Qdd_brake, Q_plan = [[] for _ in range(3)]
    if traj_type == 'orig':
        for j in range(n_q):
            Qd_plan.append(dq[j]+(c_k[j]+g_k[j]*K[j])*T_plan)
            Qdd_brake.append((-1)*Qd_plan[-1]*(1/(T_full-T_plan))) 
            Q_plan.append(q[j]+dq[j]*T_plan+0.5*(c_k[j]+g_k[j]*K[j])*T_plan**2)

    # main loop
    for t in range(n_t):
        for j in range(n_q):
            if traj_type == 'orig':
                if t < n_t_p:
                   Q[t].append(q[j]+dq[j]*T[t]+0.5*(c_k[j]+g_k[j]*K[j])*T[t]*T[t])
                else:
                   Q[t].append(Q_plan[j]+Qd_plan[j]*(T[t]-T_plan)+0.5*Qdd_brake[j]*(T[t]-T_plan)*(T[t]-T_plan))

            R_temp= gen_rotatotope_from_jrs(Q[t][-1],joint_axes[j],taylor_degree)
            R[t].append(R_temp)

    

    # throw away time/error gens and compress indep. gens to just one (if 1D) 
    if make_gens_independent:
        for t in range(n_t):
            for j in range(n_q):
                k_id = zp.conSet.PROPERTY_ID['k'][j]
                Q[t][j] = remove_dependence_and_compress(Q[t][j], k_id)
                R[t][j] = remove_dependence_and_compress(R[t][j], k_id)
        
    return Q, R



def gen_traj_JRS(q,dq,joint_axes=None,taylor_degree=1,make_gens_independent=True):

    n_q = len(q)
    if joint_axes is None:
        joint_axes = [torch.tensor([0.0,0.0,1.0]) for _ in range(n_q)]
    
    traj_type = 'orig'
    ultimate_bound = 0.0191
    k_r = 10 # Kr = kr*eye(n_q)

    T_full,T_plan, dt = 1, 0.5 ,0.01
    n_t = int(T_full/dt)
    n_t_p = int(T_plan/T_full*n_t)

    c_k = torch.zeros(n_q)
    g_k = torch.min(torch.max(PI/24,abs(dq/3)),PI/3)
    T, K = [],[]
    Q_des, Qd_des, Qdd_des, Q, Qd, Qd_a, Qdd_a, R_des, R_t_des, R, R_t = [[[] for _ in range(n_t)]for _ in range(11)]
    for _ in range(n_q):
        K.append(zp.polyZonotope([0],[[1]],prop='k'))
    for t in range(n_t):
        T.append(zp.polyZonotope([dt*t+dt/2],[[dt/2]]))
    E_p = zp.polyZonotope([0],[[ultimate_bound/k_r]])
    E_v = zp.polyZonotope([0],[[2*ultimate_bound]])

    Qd_plan, Qdd_brake, Q_plan = [[] for _ in range(3)]
    if traj_type == 'orig':
        for j in range(n_q):
            Qd_plan.append(dq[j]+(c_k[j]+g_k[j]*K[j])*T_plan)
            Qdd_brake.append((-1)*Qd_plan[-1]*(1/(T_full-T_plan))) 
            Q_plan.append(q[j]+dq[j]*T_plan+0.5*(c_k[j]+g_k[j]*K[j])*T_plan**2)

    # main loop
    for t in range(n_t):
        for j in range(n_q):
            if traj_type == 'orig':
                if t < n_t_p:
                   Q_des[t].append(q[j]+dq[j]*T[t]+0.5*(c_k[j]+g_k[j]*K[j])*T[t]*T[t])
                   Qd_des[t].append(dq[j]+(c_k[j]+g_k[j]*K[j])*T[t])
                   Qdd_des[t].append(c_k[j]+g_k[j]*K[j])
                else:
                   Q_des[t].append(Q_plan[j]+Qd_plan[j]*(T[t]-T_plan)+0.5*Qdd_brake[j]*(T[t]-T_plan)*(T[t]-T_plan))
                   Qd_des[t].append(Qd_plan[j]+Qdd_brake[j]*(T[t]-T_plan))
                   Qdd_des[t].append(Qdd_brake[j])

            Q[t].append(Q_des[t][-1]+E_p)
            Qd[t].append(Qd_des[t][-1]+E_v)
            Qd_a[t].append(Qd_des[t][-1]+k_r*E_p)
            Qdd_a[t].append(Qd_des[t][-1]+k_r*E_v)
            R_temp= gen_rotatotope_from_jrs(Q_des[t][-1],joint_axes[j],taylor_degree)
            R_des[t].append(R_temp)
            R_t_des[t].append(R_temp.T)
            R_temp = gen_rotatotope_from_jrs(Q[t][-1],joint_axes[j],taylor_degree)
            R[t].append(R_temp)
            R_t[t].append(R_temp.T)
    

    # throw away time/error gens and compress indep. gens to just one (if 1D) 
    if make_gens_independent:
        for t in range(n_t):
            for j in range(n_q):
                k_id = zp.conSet.PROPERTY_ID['k'][j]
                Q_des[t][j] = remove_dependence_and_compress(Q_des[t][j], k_id)
                Qd_des[t][j] = remove_dependence_and_compress(Qd_des[t][j], k_id)
                Qdd_des[t][j] = remove_dependence_and_compress(Qdd_des[t][j], k_id)
                Q[t][j] = remove_dependence_and_compress(Q[t][j], k_id)
                Qd[t][j] = remove_dependence_and_compress(Qd[t][j], k_id)
                Qd_a[t][j] = remove_dependence_and_compress(Qd_a[t][j], k_id)
                Qdd_a[t][j] = remove_dependence_and_compress(Qdd_a[t][j], k_id)
                R_des[t][j] = remove_dependence_and_compress(R_des[t][j], k_id)
                R_t_des[t][j] = remove_dependence_and_compress(R_t_des[t][j], k_id)
                R[t][j] = remove_dependence_and_compress(R[t][j], k_id)
                R_t[t][j] = remove_dependence_and_compress(R_t[t][j], k_id)
    #import pdb; pdb.set_trace()
    return Q_des, Qd_des, Qdd_des, Q, Qd, Qd_a, Qdd_a, R_des, R_t_des, R, R_t





if __name__ == '__main__':
    q = torch.tensor([0])
    dq = torch.tensor([torch.pi])
    Q,R = gen_JRS(q,dq,joint_axes=None,taylor_degree=1,make_gens_independent=False)
    import pdb; pdb.set_trace()
    

    PZ_JRS,_ = zp.load_JRS_trig(q,dq)
    ax = zp.plot_polyzonos(PZ_JRS,plot_freq=1,edgecolor='blue',hold_on=True)
    #zp.plot_polyzonos(PZ_JRS,plot_freq=1,ax=ax)
    
    zp.plot_JRSs(Q,deg=1,plot_freq=1,ax=ax)
    
    
    