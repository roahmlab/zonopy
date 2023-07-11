import torch
import numpy as np
import zonopy.robots2.robot as robots2
import zonopy.transformations.rotation as rot
from zonopy.joint_reachable_set.gen_jrs import JrsGenerator
import zonopy as zp

robots2.DEBUG_VIZ = False
print('Loading Robot')
# This is hardcoded for now
rob = robots2.load_robot('/data/git/zonopy/zonopy/robots/assets/robots/kinova_arm/gen3.urdf')
q = np.array([0.624819195837238,-1.17185521197975,-2.04687142485692,1.69686054456768,-2.28521956398477,0.151194251967712,1.54233217035569])
qd = np.array([-0.0218762290685389,-0.0972760750895341,0.118467026460654,0.00255072010498519,0.118466729140505,-0.118467364612488,-0.0533775122637854])
qdd = np.array([0.0249296393119391,0.110843270840544,-0.133003332695036,-0.00290896919579042,-0.133005741757336,0.133000561712863,0.0608503609673116])

# Set cuda if available
# zp.setup_cuda()

print('Staring JRS Generation')
# traj_class=zp.trajectories.BernsteinArmTrajectory
traj_class=zp.trajectories.PiecewiseArmTrajectory
a = JrsGenerator(rob, traj_class=traj_class, ultimate_bound=0.0191, k_r=10)
b = a.gen_JRS(q, qd, qdd)
# c = JrsGenerator._get_pz_rotations_from_q(b[0][0][0],a.joint_axis[0],taylor_deg=1)

print('pause')

