import unittest
import os
import zonopy as zp
import zonopy.robots2.robot as robots2
from zonopy import JrsGenerator
import numpy as np
from zonopy.kinematics.SO import sphere_occupancy, make_spheres
import torch
import time

# Test cases to make usre the the forward spherical occupancy is numerically correct
class TestFSO(unittest.TestCase):
    def setUp(self):
        basedirname = os.path.dirname(zp.__file__)
        # This specific file is important! Numbers will need to change if it changes.
        self.robot = robots2.ZonoArmRobot.load(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'), create_joint_occupancy=True)
        # initial conditions
        self.q = np.array([0.624819195837238,-1.17185521197975,-2.04687142485692,1.69686054456768,-2.28521956398477,0.151194251967712,1.54233217035569])
        self.qd = np.array([-0.0218762290685389,-0.0972760750895341,0.118467026460654,0.00255072010498519,0.118466729140505,-0.118467364612488,-0.0533775122637854])
        self.qdd = np.array([0.0249296393119391,0.110843270840544,-0.133003332695036,-0.00290896919579042,-0.133005741757336,0.133000561712863,0.0608503609673116])

        traj_class=zp.trajectories.BernsteinArmTrajectory
        self.jrsgen = JrsGenerator(self.robot, traj_class=traj_class, ultimate_bound=0.0191, k_r=10)
        self.batchjrsgen = JrsGenerator(self.robot, traj_class=traj_class, ultimate_bound=0.0191, k_r=10, batched=True, unique_tid=False)
        self.jrs = self.jrsgen.gen_JRS(self.q, self.qd, self.qdd)
        self.batchjrs = self.batchjrsgen.gen_JRS(self.q, self.qd, self.qdd)

    # At this point in time, just check if they gradients match up and if it funcionally runs
    # Later, can test against known values
    def test_serial_operation(self):
        test_k_list = [torch.zeros(7), torch.zeros(7) + 0.1, torch.linspace(0, 1, 7)]
        n_spheres_list = [4, 5, 6]

        jso_serial = []
        for Rs in self.jrs['R']:
            jso, fso_pairs, _ = sphere_occupancy(Rs, self.robot)
            jso_serial.append(jso)

        joint_pairs = []
        for pairs in fso_pairs.values():
            joint_pairs.extend(pairs)

        def test_grad_fcn(k, j1, j2, t_idx, n_spheres=5):
            p1 = jso_serial[t_idx][j1][0].center_slice_all_dep(k)
            p2 = jso_serial[t_idx][j2][0].center_slice_all_dep(k)
            r1 = jso_serial[t_idx][j1][1]
            r2 = jso_serial[t_idx][j2][1]
            return make_spheres(p1, p2, r1, r2, n_spheres=n_spheres)
        
        def get_grad_fcn(k, j1, j2, t_idx, n_spheres=5):
            p1 = jso_serial[t_idx][j1][0].center_slice_all_dep(k)
            p2 = jso_serial[t_idx][j2][0].center_slice_all_dep(k)
            jac1 = jso_serial[t_idx][j1][0].grad_center_slice_all_dep(k)
            jac2 = jso_serial[t_idx][j2][0].grad_center_slice_all_dep(k)
            r1 = jso_serial[t_idx][j1][1]
            r2 = jso_serial[t_idx][j2][1]
            _, _, jacc, jacr = make_spheres(p1, p2, r1, r2, center_jac_1=jac1, center_jac_2=jac2, n_spheres=n_spheres)
            return (jacc, jacr)
        
        for k in test_k_list:
            for n_spheres in n_spheres_list:
                for t_idx in range(len(jso_serial)):
                    for pair in joint_pairs:
                        test_grad = lambda c: test_grad_fcn(c, pair[0], pair[1], t_idx, n_spheres=n_spheres)

                        out = torch.autograd.functional.jacobian(test_grad, k)
                        comp = get_grad_fcn(k, pair[0], pair[1], t_idx, n_spheres=n_spheres)

                        self.assertTrue(torch.allclose(out[0], comp[0]))
                        self.assertTrue(torch.allclose(out[1], comp[1]))
    
    def test_batch_operation(self):
        test_k_list = [torch.zeros(7), torch.zeros(7) + 0.1, torch.linspace(0, 1, 7)]
        n_spheres_list = [4, 5, 6]

        jso, fso_pairs, _ = sphere_occupancy(self.batchjrs['R'], self.robot)

        joint_pairs = []
        for pairs in fso_pairs.values():
            joint_pairs.extend(pairs)

        def test_grad_fcn(k, j1, j2, n_spheres=5):
            p1 = jso[j1][0].center_slice_all_dep(k)
            p2 = jso[j2][0].center_slice_all_dep(k)
            r1 = jso[j1][1]
            r2 = jso[j2][1]
            return make_spheres(p1, p2, r1, r2, n_spheres=n_spheres)
        
        def get_grad_fcn(k, j1, j2, n_spheres=5):
            p1 = jso[j1][0].center_slice_all_dep(k)
            p2 = jso[j2][0].center_slice_all_dep(k)
            jac1 = jso[j1][0].grad_center_slice_all_dep(k)
            jac2 = jso[j2][0].grad_center_slice_all_dep(k)
            r1 = jso[j1][1]
            r2 = jso[j2][1]
            _, _, jacc, jacr = make_spheres(p1, p2, r1, r2, center_jac_1=jac1, center_jac_2=jac2, n_spheres=n_spheres)
            return (jacc, jacr)
        
        for k in test_k_list:
            for n_spheres in n_spheres_list:
                for pair in joint_pairs:
                    test_grad = lambda c: test_grad_fcn(c, pair[0], pair[1], n_spheres=n_spheres)

                    out = torch.autograd.functional.jacobian(test_grad, k)
                    comp = get_grad_fcn(k, pair[0], pair[1], n_spheres=n_spheres)

                    self.assertTrue(torch.allclose(out[0], comp[0]))
                    self.assertTrue(torch.allclose(out[1], comp[1]))
    
    def test_full_SFO(self):
        n_k = 7
        k = torch.zeros(n_k)

        t1 = time.perf_counter()
        jso, fso_pairs, _ = sphere_occupancy(self.batchjrs['R'], self.robot)
        print("Time to generate initial occupancy", time.perf_counter()-t1)
        
        t1 = time.perf_counter()
        joint_pairs = []
        for pairs in fso_pairs.values():
            joint_pairs.extend(pairs)
        
        n_joints = len(jso)
        n_pairs = len(joint_pairs)
        n_spheres = 5
        n_time = 100
        total_spheres = n_joints*n_time + n_spheres*n_pairs*n_time

        centers = torch.empty((total_spheres, 3))
        radii = torch.empty((total_spheres))
        center_jac = torch.empty((total_spheres, 3, n_k))
        radii_jac = torch.empty((total_spheres, n_k))

        # First process with all the joints
        joints = {}
        for j, (name, (pz, r)) in enumerate(jso.items()):
            sidx = j*n_time
            eidx = (j+1)*n_time
            centers[sidx:eidx] = pz.center_slice_all_dep(k)
            center_jac[sidx:eidx] = pz.grad_center_slice_all_dep(k)
            radii[sidx:eidx] = r
            radii_jac[sidx:eidx] = 0
            joints[name] = (centers[sidx:eidx], r, center_jac[sidx:eidx])

        # Now process all inbetweens
        for j in range(n_pairs):
            joint1, joint2 = joint_pairs[j]
            p1, r1, jac1 = joints[joint1]
            p2, r2, jac2 = joints[joint2]
            sidx = n_joints*n_time + j*n_spheres*n_time
            eidx = n_joints*n_time + (j+1)*n_spheres*n_time
            spheres = make_spheres(p1, p2, r1, r2, jac1, jac2, n_spheres)
            centers[sidx:eidx] = spheres[0].reshape(-1,3)
            radii[sidx:eidx] = spheres[1].reshape(-1)
            center_jac[sidx:eidx] = spheres[2].reshape(-1,3,n_k)
            radii_jac[sidx:eidx] = spheres[3].reshape(-1,n_k)
        
        print("Time to generate sphere occupancy for slice", time.perf_counter()-t1)

if __name__ == '__main__':
    unittest.main()
