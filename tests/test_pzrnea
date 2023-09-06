import numpy as np
import torch
import zonopy as zp
import unittest
import zonopy.robots2.robot as robots2
from zonopy.joint_reachable_set.gen_jrs import JrsGenerator
from zonopy.dynamics.RNEA import pzrnea
import os

# Test cases to make sure PZRNEA is behaving as expected. They're specified with hard values relative to a specific file.
class TestPZRNEA(unittest.TestCase):
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
    
    def test_batch_singular_equality(self):
        out_batch = pzrnea(self.batchjrs['R'], self.batchjrs['qd'], self.batchjrs['qd_aux'], self.batchjrs['qdd_aux'], self.robot)
        out_0 = pzrnea(self.jrs['R'][0], self.jrs['qd'][0], self.jrs['qd_aux'][0], self.jrs['qdd_aux'][0], self.robot)
        out_20 = pzrnea(self.jrs['R'][20], self.jrs['qd'][20], self.jrs['qd_aux'][20], self.jrs['qdd_aux'][20], self.robot)
        out_75 = pzrnea(self.jrs['R'][75], self.jrs['qd'][75], self.jrs['qd_aux'][75], self.jrs['qdd_aux'][75], self.robot)
        out_99 = pzrnea(self.jrs['R'][99], self.jrs['qd'][99], self.jrs['qd_aux'][99], self.jrs['qdd_aux'][99], self.robot)

        def test_compare(batch_dict, idx, single_dict):
            # we compare the resulting intervals as they should be the same even after reduction
            def key_interval_check(key, bjoint_dict, joint_dict):
                if key in bjoint_dict or key in joint_dict:
                    b_interval = bjoint_dict[key][idx].to_interval()
                    interval = joint_dict[key].to_interval()
                    self.assertTrue(torch.allclose(b_interval.inf, interval.inf))
                    self.assertTrue(torch.allclose(b_interval.sup, interval.sup))
            # Go through expected keys
            for bjoint_dict, joint_dict in zip(batch_dict.values(), single_dict.values()):
                key_interval_check('torque', bjoint_dict, joint_dict)
                key_interval_check('force', bjoint_dict, joint_dict)
                key_interval_check('moment', bjoint_dict, joint_dict)
        
        test_compare(out_batch, 0, out_0)
        test_compare(out_batch, 20, out_20)
        test_compare(out_batch, 75, out_75)
        test_compare(out_batch, 99, out_99)

    def test_known_centers(self):
        center_R = [zp.batchMatPolyZonotope(R.C.unsqueeze(1)) for R in self.batchjrs['R']]
        center_qd = [zp.batchPolyZonotope(qd.c.unsqueeze(1)) for qd in self.batchjrs['qd']]
        center_qd_aux = [zp.batchPolyZonotope(qd_aux.c.unsqueeze(1)) for qd_aux in self.batchjrs['qd_aux']]
        center_qdd_aux = [zp.batchPolyZonotope(qdd_aux.c.unsqueeze(1)) for qdd_aux in self.batchjrs['qdd_aux']]
        out_batch = pzrnea(center_R, center_qd, center_qd_aux, center_qdd_aux, self.robot)

        torques_0_24_83_99 = np.array(
            [[-2.2043249e-02, -4.2815480e-02,  2.3025298e-02,  3.2934831e-03],
             [ 1.5531300e+01,  1.5579023e+01,  1.5306368e+01,  1.5348853e+01],
             [ 5.8900938e+00,  5.9691648e+00,  5.8428183e+00,  5.8228750e+00],
             [-2.3977411e+00, -2.3162429e+00, -2.3437095e+00, -2.3652601e+00],
             [-6.8723798e-02, -5.3948201e-02, -6.5783419e-02, -6.7633063e-02],
             [-3.4169534e-01, -3.7699845e-01, -3.5043979e-01, -3.4166098e-01],
             [ 4.4905085e-02,  4.4376034e-02,  4.4783697e-02,  4.4622652e-02]],
        ) # n_joints by time_idx's

        torques = [out_dict['torque'][[0, 24, 83, 99]] for out_dict in list(out_batch.values())[:-1]]
        torques = np.array([val.c.numpy() for val in torques]).squeeze()
        self.assertTrue(np.allclose(torques, torques_0_24_83_99))

if __name__ == '__main__':
    unittest.main()
