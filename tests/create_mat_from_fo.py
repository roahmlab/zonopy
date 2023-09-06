import torch
import numpy as np
import zonopy.robots2.robot as robots2
from scipy.io import savemat
from zonopy.joint_reachable_set.gen_jrs import JrsGenerator
import zonopy as zp
import zonopy.kinematics as kin
import os
basedirname = os.path.dirname(zp.__file__)


# Settings
robot_file = 'robots/assets/robots/kinova_arm/gen3.urdf'
save_base_locname = 'export'
use_cuda = False
use_bernstein = True
save_fo = True
save_jo = True
save_jobz = True
reduce = True
only_generate_R_online = True

q = np.array([0.624819195837238,-1.17185521197975,-2.04687142485692,1.69686054456768,-2.28521956398477,0.151194251967712,1.54233217035569])
qd = np.array([-0.0218762290685389,-0.0972760750895341,0.118467026460654,0.00255072010498519,0.118466729140505,-0.118467364612488,-0.0533775122637854])
qdd = np.array([0.0249296393119391,0.110843270840544,-0.133003332695036,-0.00290896919579042,-0.133005741757336,0.133000561712863,0.0608503609673116])
k_slice = np.array([.5, .5, .5, .5, .5, .5, .5])

#####

# Set cuda if desired and available
if use_cuda:
    zp.setup_cuda()

# Load robot
print('Loading Robot')
# make sure not to show any debug vizualizations
robots2.DEBUG_VIZ = False
rob = robots2.ZonoArmRobot.load(os.path.join(basedirname, robot_file), create_joint_occupancy=True)


print('Starting JRS Generation')
traj_class=zp.trajectories.BernsteinArmTrajectory if use_bernstein else zp.trajectories.PiecewiseArmTrajectory
generator = JrsGenerator(rob, traj_class=traj_class, ultimate_bound=0.0191, k_r=10, batched=True, unique_tid=False)
jrs_out = generator.gen_JRS(q, qd, qdd, only_R=only_generate_R_online)
print('Finished JRS Generation')

if only_generate_R_online:
    joints = jrs_out
else:
    joints = jrs_out['R']

print("Perform forward operations")
# fk = kin.forward_kinematics(joints, rob.robot)
fo, _ = kin.forward_occupancy(joints, rob)
jo, _ = kin.joint_occupancy(joints, rob)
jobz, _ = kin.joint_occupancy(joints, rob, use_outer_bb=True)


print("Exporting")
# Process out the forward occupancy (Assume only 1 batch dim of time)
def process_out(in_dict):
    out_rec = np.recarray((1,), dtype=[(n,object) for n in in_dict.keys()])
    for link_name, occupancy in in_dict.items():
        try:
            occupancy = occupancy.slice_all_dep(torch.as_tensor(k_slice, dtype=torch.get_default_dtype()).view(1,generator.num_q).repeat(generator.num_t,1))
            ent = np.empty(generator.num_t, dtype=object)
            for i in range(generator.num_t):
                if reduce:
                    ent[i] = (occupancy[i].reduce(4).Z).cpu().numpy().T
                else:
                    ent[i] = (occupancy[i].Z).cpu().numpy().T
        except:
            occupancy = occupancy.slice_all_dep(torch.as_tensor(k_slice, dtype=torch.get_default_dtype()))
            if reduce:
                ent = (occupancy.reduce(4).Z).cpu().numpy().T
            else:
                ent = (occupancy.Z).cpu().numpy().T
        out_rec[link_name][0] = ent

    return out_rec

export = {}
name = '_reduced' if reduce else ''
if save_fo:
    export['fo'] = process_out(fo)
    name += '_fo'
if save_jo:
    export['jo'] = process_out(jo)
    name += '_jo'
if save_jobz:
    export['jobz'] = process_out(jobz)
    name += '_jobz'

print("Saving to", save_base_locname + name + '.mat')
savemat(save_base_locname + name + '.mat', export, do_compression=True)