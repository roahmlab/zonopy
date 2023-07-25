import torch
import numpy as np
import zonopy.robots2.robot as robots2
import zonopy.transformations.rotation as rot
from zonopy.joint_reachable_set.gen_jrs import JrsGenerator
import zonopy as zp
import zonopy.kinematics as kin

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
traj_class=zp.trajectories.BernsteinArmTrajectory
# traj_class=zp.trajectories.PiecewiseArmTrajectory
a = JrsGenerator(rob, traj_class=traj_class, ultimate_bound=0.0191, k_r=10, unique_tid=False)
b = a.gen_JRS(q, qd, qdd)
print('Finished JRS Generation')


# Do serial forward operations
serial_forward = []
for Rs in b['R']:
    Rs = list(Rs)
    fk = kin.forward_kinematics(Rs, rob.robot)
    fo, _ = kin.forward_occupancy(Rs, rob.robot)
    jo, _ = kin.joint_occupancy(Rs, rob.robot)
    jo_bzlike, _ = kin.joint_occupancy(Rs, rob.robot, use_outer_bb=True)
    serial_forward.append((fk, fo, jo, jo_bzlike))


# Do batched forward operations
print('Testing batched JRS Generation')
c = JrsGenerator(rob, traj_class=traj_class, ultimate_bound=0.0191, k_r=10, batched=True, unique_tid=False)
d = c.gen_JRS(q, qd, qdd)
print('Finished batched JRS Generation')
joints = list(d['R'])

fk = kin.forward_kinematics(joints, rob.robot)
fo, _ = kin.forward_occupancy(joints, rob.robot)
jo, _ = kin.joint_occupancy(joints, rob.robot)
jo_bzlike, _ = kin.joint_occupancy(joints, rob.robot, use_outer_bb=True)


# Plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as a3

# Get reference
plot_arm = False
plot_ref = True
slice_zonos = True
plot_seperate = True
k = np.ones(7)*0.5
if plot_arm and plot_ref:
    reference, _, _ = traj_class(q, qd, qdd, k, a.param_range).getReference(np.arange(0, 1, 0.01))

if plot_arm and plot_seperate:
    fig = plt.figure(figsize=(21,9))
    subfigs = fig.subfigures(1, 3, wspace=0.07)
    ax1 = a3.Axes3D(subfigs[0])
    ax2 = a3.Axes3D(subfigs[1])
    ax3 = a3.Axes3D(subfigs[2])
    ax3.set_xlim([-.55,0.25])
    ax3.set_ylim([-.1,0.7])
    ax3.set_zlim([0,0.8])
else:
    fig = plt.figure(figsize=(14,9))
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    ax1 = a3.Axes3D(subfigs[0])
    ax2 = a3.Axes3D(subfigs[1])

if plot_arm and not plot_ref:
    ref_fk = rob.robot.visual_trimesh_fk(cfg=q)
    for mesh, transform in ref_fk.items():
        mesh = mesh.copy().apply_transform(transform)
        if plot_seperate:
            ax3.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces,color='orange')
        else:
            ax1.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces,color='orange')
            ax2.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces,color='orange')

for i in range(100):
    print(i)
    def create_patches(body_dict):
        patches = []
        for occupancy in body_dict.values():
            if slice_zonos:
                occupancy = occupancy.slice_all_dep(torch.as_tensor(k, dtype=torch.get_default_dtype()))
            else:
                occupancy = occupancy.to_zonotope()
            patch = occupancy.reduce(4).polyhedron_patch(ax1)
            patches.extend(patch)
        return patches
    
    serial_fo = create_patches(serial_forward[i][1])
    serial_jo = create_patches(serial_forward[i][2])
    serial_jobz = create_patches(serial_forward[i][3])

    serial_fo = Poly3DCollection(serial_fo,alpha=0.1,edgecolor='red',facecolor='red',linewidths=0.5)
    serial_jo = Poly3DCollection(serial_jo,alpha=0.1,edgecolor='green',facecolor='green',linewidths=0.5)
    serial_jobz = Poly3DCollection(serial_jobz,alpha=0.1,edgecolor='blue',facecolor='blue',linewidths=0.5)

    ax1.add_collection3d(serial_fo)
    ax1.add_collection3d(serial_jo)
    ax1.add_collection3d(serial_jobz)
    ax1.set_xlim([-.55,0.25])
    ax1.set_ylim([-.1,0.7])
    ax1.set_zlim([0,0.8])

    def create_patches_batch(batch_dict, i):
        patches = []
        for occupancy in batch_dict.values():
            try:
                occupancy = occupancy[i]
            except:
                occupancy = occupancy
            if slice_zonos:
                occupancy = occupancy.slice_all_dep(torch.as_tensor(k, dtype=torch.get_default_dtype()))
                # occupancy = occupancy.slice_all_dep(torch.as_tensor(k, dtype=torch.get_default_dtype()).view(1,7).repeat(100,1))
            else:
                occupancy = occupancy.to_zonotope()
            patch = occupancy.reduce(4).polyhedron_patch(ax2)
            patches.extend(patch)
        return patches
    
    batch_fo = create_patches_batch(fo,i)
    batch_jo = create_patches_batch(jo,i)
    batch_jobz = create_patches_batch(jo_bzlike,i)

    batch_fo = Poly3DCollection(batch_fo,alpha=0.1,edgecolor='red',facecolor='red',linewidths=0.5)
    batch_jo = Poly3DCollection(batch_jo,alpha=0.1,edgecolor='green',facecolor='green',linewidths=0.5)
    batch_jobz = Poly3DCollection(batch_jobz,alpha=0.1,edgecolor='blue',facecolor='blue',linewidths=0.5)

    ax2.add_collection3d(batch_fo)
    ax2.add_collection3d(batch_jo)
    ax2.add_collection3d(batch_jobz)
    ax2.set_xlim([-.55,0.25])
    ax2.set_ylim([-.1,0.7])
    ax2.set_zlim([0,0.8])

    ref1=[]
    ref2=[]
    ref3=[]
    if plot_arm and plot_ref:
        ref_fk = rob.robot.visual_trimesh_fk(cfg=reference[i])
        for mesh, transform in ref_fk.items():
            mesh = mesh.copy().apply_transform(transform)
            if plot_seperate:
                ref3.append(ax3.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces,color='orange'))
            else:
                ref1.append(ax1.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces,color='orange'))
                ref2.append(ax2.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces,color='orange'))

    plt.draw()
    plt.pause(0.1)
    plt.savefig(f'render/compc{i:02d}.png', bbox_inches='tight')
    serial_fo.remove()
    serial_jo.remove()
    serial_jobz.remove()
    batch_fo.remove()
    batch_jo.remove()
    batch_jobz.remove()
    for ref in ref1: ref.remove()
    for ref in ref2: ref.remove()
    for ref in ref3: ref.remove()
    # ax1.clear()
    # ax2.clear()
    # if plot_ref:
        # ax3.clear()

# plt.show()