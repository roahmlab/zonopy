import torch
import numpy as np
import zonopy.robots2.robot as robots2
import zonopy.transformations.rotation as rot
from zonopy.joint_reachable_set.gen_jrs import JrsGenerator
import zonopy as zp


# Set cuda if desired and available
use_cuda = False
if use_cuda:
    zp.setup_cuda()


# Load robot
import os
basedirname = os.path.dirname(zp.__file__)

robots2.DEBUG_VIZ = False
print('Loading Robot')
# This is hardcoded for now
rob = robots2.ArmRobot(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'))
q = np.array([0.624819195837238,-1.17185521197975,-2.04687142485692,1.69686054456768,-2.28521956398477,0.151194251967712,1.54233217035569])
qd = np.array([-0.0218762290685389,-0.0972760750895341,0.118467026460654,0.00255072010498519,0.118466729140505,-0.118467364612488,-0.0533775122637854])
qdd = np.array([0.0249296393119391,0.110843270840544,-0.133003332695036,-0.00290896919579042,-0.133005741757336,0.133000561712863,0.0608503609673116])


print('Starting JRS Generation')
# traj_class=zp.trajectories.BernsteinArmTrajectory
traj_class=zp.trajectories.PiecewiseArmTrajectory
a = JrsGenerator(rob, traj_class=traj_class, ultimate_bound=0.0191, k_r=10)
b = a.gen_JRS(q, qd, qdd)
print('Finished JRS Generation')
# c = JrsGenerator._get_pz_rotations_from_q(b[0][0][0],a.joint_axis[0],taylor_deg=1)

print('Testing kinematics')
import zonopy.kinematics as kin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as a3
fig = plt.figure()
# ax = fig.gca()
ax = a3.Axes3D(fig)
for i in range (0,100,10):
    patches = []
    print('t', i)
    fk = kin.forward_kinematics(list(b['R'][i]), rob.robot)
    print('updating plot')
    for name, (pos, rot) in fk.items():
        if name in ['base_link']:#,'shoulder_link','half_arm_1_link']:
            continue
        bounds = torch.as_tensor(rob.robot.link_parent_joint[name].radius)
        pos = pos + rot@zp.polyZonotope(torch.vstack([torch.zeros(3), torch.diag(bounds)]))
        patch = pos.to_zonotope().reduce(4).polyhedron_patch(ax)
        patches.extend(patch)
    patches = Poly3DCollection(patches,alpha=0.03,edgecolor='green',facecolor='green',linewidths=0.5)
    ax.add_collection3d(patches)  
    # plt.autoscale()
    plt.draw()
    plt.pause(0.1)
# Combine all the R for a joint into one batch mat poly zono
# joints = []
# for joint_Rs in b['R'].T:
#     # Assume these all have the same expMat and id's!
#     # This only true for a single given call to JrsGenerator
#     Z = torch.stack([zono.Z for zono in joint_Rs])
#     batch_zono = zp.batchMatPolyZonotope(Z, joint_Rs[0].n_dep_gens, joint_Rs[0].expMat, joint_Rs[0].id, compress=0)
#     joints.append(batch_zono)

# kin.forward_kinematics(joints, rob.robot)

plt.show()

# Timing
num = 10
print("Start Timing", num, "Loops")
import timeit
duration = timeit.timeit(lambda: JrsGenerator(rob, traj_class=traj_class, ultimate_bound=0.0191, k_r=10).gen_JRS(q, qd, qdd), number=num)
print('Took', duration/num, 'seconds each loop for', num, 'loops')

# Profiling
# from torch.profiler import profile, record_function, ProfilerActivity
# with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True, use_cuda=use_cuda) as prof:
#     with record_function("create_jrs"):
#         JrsGenerator(rob, traj_class=traj_class, ultimate_bound=0.0191, k_r=10).gen_JRS(q, qd, qdd)
# prof.export_chrome_trace("trace.json")

print('pause')

