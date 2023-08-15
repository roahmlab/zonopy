import numpy as np
import torch
import zonopy as zp
from zonopy.conSet.polynomial_zonotope.utils import remove_dependence_and_compress
import unittest

# Helper
rem_dep = np.vectorize(remove_dependence_and_compress, excluded=[1])

# Test cases to just make sure the trajectory is likely valid and matches expectation
# for the original ArmTD formulation using polynomial zonotopes
class TestPiecewiseArmTrajectories(unittest.TestCase):
    def setUp(self):
        # initial conditions
        self.q = np.array([0, 0, 0])
        self.dq = np.array([np.pi, np.pi/2, np.pi/4])
        tdiscretization = 0.01
        duration = 1.

        # Params
        self.params = np.array([
            zp.polyZonotope([[0],[1]],1,id=0),
            zp.polyZonotope([[0],[1]],1,id=1),
            zp.polyZonotope([[0],[1]],1,id=2)
            ])
        self.param_range = np.minimum(np.maximum(np.ones(3) * np.pi/24, np.abs(self.dq/3)), np.ones(3) * np.pi/3)
        self.param_range = np.vstack([-self.param_range, self.param_range])

        # Time discretization
        i_s = np.arange(int(duration/tdiscretization))
        gens = torch.ones(100) * tdiscretization/2
        centers = torch.as_tensor(tdiscretization*i_s+tdiscretization/2, dtype=torch.get_default_dtype())
        z = torch.vstack([centers, gens]).unsqueeze(2).transpose(0,1)
        self.times = zp.batchPolyZonotope(z, 1, id=3)
    
    def test_construction3link(self):
        zp.trajectories.PiecewiseArmTrajectory(self.q, self.dq, None, self.params, self.param_range, tbrake=0.5, tfinal=1)

    def test_generation3link(self):
        gen = zp.trajectories.PiecewiseArmTrajectory(self.q, self.dq, None, self.params, self.param_range, tbrake=0.5, tfinal=1)
        out = gen.getReference(np.array([self.times[i] for i in range(self.times.batch_shape[0])]))
        
        # Check outputs
        self.assertEqual(len(out), 3, "Expected tuple of 3 on output from getReference")
        self.assertEqual(len(out[0].T), 3, "Expected three joints on output from getReference")
        self.assertEqual(len(out[0].T[0]), 100, "Expected an output size of 100 for position output")
        self.assertEqual(len(out[1].T[0]), 100, "Expected an output size of 100 for velocity output")
        self.assertEqual(len(out[2].T[0]), 100, "Expected an output size of 100 for acceleration output")

        # Check ID's
        self.assertTrue(np.all(np.sort(out[1][0][0].id) == np.array([0,3])), "Expected id 0 and 3 for joint 0")
        self.assertTrue(np.all(np.sort(out[1][0][1].id) == np.array([1,3])), "Expected id 1 and 3 for joint 1")
        self.assertTrue(np.all(np.sort(out[1][0][2].id) == np.array([2,3])), "Expected id 2 and 3 for joint 2")

        # Remove dependence and compress
        Q_all = rem_dep(out[0], [0, 1, 2])
        Qd_all = rem_dep(out[1], [0, 1, 2])
        Qdd_all = rem_dep(out[2], [0, 1, 2])

        ## For the first link
        Q = Q_all.T[0]
        Qd = Qd_all.T[0]
        Qdd = Qdd_all.T[0]
        # Test initial and final values
        q0_Z = torch.tensor(
            [[0.0157079632679490],
            [1.30899693899575e-05],
            [0.0157472331761188]]
        )
        qd0_Z = torch.tensor(
            [[3.14159265358979],
            [0.00523598775598299],
            [0.00523598775598299]]
        )
        q99_Z = torch.tensor(
            [[2.35611595037601],
            [0.261786297829759],
            [0.000274889357189107]]
        )
        qd99_Z = torch.tensor(
            [[0.0314159265358978],
            [0.00523598775598300],
            [0.0366519142918809]]
        )
        self.assertTrue(torch.allclose(q0_Z, Q[0].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(qd0_Z, Qd[0].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(q99_Z, Q[99].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(qd99_Z, Qd[99].reduce_indep(1).Z))

        # Test acceleration
        qdd_up_Z = torch.tensor(
            [[0],
            [1.04719755119660]]
        )
        qdd_down_Z = torch.tensor(
            [[-6.28318530717959],
            [-1.04719755119660]]
        )
        for i in range(50):
            self.assertTrue(torch.allclose(qdd_up_Z, Qdd[0+i].reduce_indep(1).Z))
            self.assertTrue(torch.allclose(qdd_down_Z, Qdd[50+i].reduce_indep(1).Z))

        # Test 3 random values
        q8_Z = torch.tensor(
            [[0.267035375555132],
            [0.00378300115369771],
            [0.0161661121965975]]
        )
        qd8_Z = torch.tensor(
            [[3.14159265358979],
            [0.0890117918517108],
            [0.00523598775598299]]
        )

        q21_Z = torch.tensor(
            [[0.675442420521806],
            [0.0242033534020314],
            [0.0168467906048753]]
        )
        qd21_Z = torch.tensor(
            [[3.14159265358979],
            [0.225147473507269],
            [0.00523598775598299]]
        )

        q79_Z = torch.tensor(
            [[2.22416905892523],
            [0.239795149254631],
            [0.00760527221556529]]
        )
        qd79_Z = torch.tensor(
            [[1.28805298797182],
            [0.214675497995303],
            [0.0366519142918809]]
        )
        self.assertTrue(torch.allclose(q8_Z, Q[8].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(qd8_Z, Qd[8].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(q21_Z, Q[21].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(qd21_Z, Qd[21].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(q79_Z, Q[79].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(qd79_Z, Qd[79].reduce_indep(1).Z))

    def test_generation3link_batch(self):
        gen = zp.trajectories.PiecewiseArmTrajectory(self.q, self.dq, None, self.params, self.param_range)
        out = gen.getReference(self.times)
        
        # Check outputs
        self.assertEqual(len(out), 3, "Expected tuple of 3 on output from getReference")
        self.assertEqual(len(out[0]), 3, "Expected three joints on output from getReference")
        self.assertEqual(out[0][0].batch_shape[0], 100, "Expected a batch size of 100 for position output")
        self.assertEqual(out[1][0].batch_shape[0], 100, "Expected a batch size of 100 for velocity output")
        self.assertEqual(out[2][0].batch_shape[0], 100, "Expected a batch size of 100 for acceleration output")

        # Check ID's
        self.assertTrue(np.all(np.sort(out[1][0].id) == np.array([0,3])), "Expected id 0 and 3 for joint 0")
        self.assertTrue(np.all(np.sort(out[1][1].id) == np.array([1,3])), "Expected id 1 and 3 for joint 1")
        self.assertTrue(np.all(np.sort(out[1][2].id) == np.array([2,3])), "Expected id 2 and 3 for joint 2")

        # Remove dependence and compress
        Q_all = rem_dep(out[0], [0, 1, 2])
        Qd_all = rem_dep(out[1], [0, 1, 2])
        Qdd_all = rem_dep(out[2], [0, 1, 2])

        ## For the first link
        Q = Q_all[0].reduce_indep(1)
        Qd = Qd_all[0].reduce_indep(1)
        Qdd = Qdd_all[0].reduce_indep(1)
        # Test initial and final values
        q0_Z = torch.tensor(
            [[0.0157079632679490],
            [1.30899693899575e-05],
            [0.0157472331761188]]
        )
        qd0_Z = torch.tensor(
            [[3.14159265358979],
            [0.00523598775598299],
            [0.00523598775598299]]
        )
        q99_Z = torch.tensor(
            [[2.35611595037601],
            [0.261786297829759],
            [0.000274889357189107]]
        )
        qd99_Z = torch.tensor(
            [[0.0314159265358978],
            [0.00523598775598300],
            [0.0366519142918809]]
        )
        self.assertTrue(torch.allclose(q0_Z, Q[0].Z))
        self.assertTrue(torch.allclose(qd0_Z, Qd[0].Z))
        self.assertTrue(torch.allclose(q99_Z, Q[99].Z))
        self.assertTrue(torch.allclose(qd99_Z, Qd[99].Z))

        # Test all acceleration
        qdd_up_Z = torch.tensor(
            [[0],
            [1.04719755119660]]
        )
        qdd_down_Z = torch.tensor(
            [[-6.28318530717959],
            [-1.04719755119660]]
        )
        self.assertTrue(torch.allclose(qdd_up_Z.expand_as(Qdd[:50].Z), Qdd[:50].Z))
        self.assertTrue(torch.allclose(qdd_down_Z.expand_as(Qdd[50:].Z), Qdd[50:].Z))

        # Test 3 random values
        q8_Z = torch.tensor(
            [[0.267035375555132],
            [0.00378300115369771],
            [0.0161661121965975]]
        )
        qd8_Z = torch.tensor(
            [[3.14159265358979],
            [0.0890117918517108],
            [0.00523598775598299]]
        )

        q21_Z = torch.tensor(
            [[0.675442420521806],
            [0.0242033534020314],
            [0.0168467906048753]]
        )
        qd21_Z = torch.tensor(
            [[3.14159265358979],
            [0.225147473507269],
            [0.00523598775598299]]
        )

        q79_Z = torch.tensor(
            [[2.22416905892523],
            [0.239795149254631],
            [0.00760527221556529]]
        )
        qd79_Z = torch.tensor(
            [[1.28805298797182],
            [0.214675497995303],
            [0.0366519142918809]]
        )
        self.assertTrue(torch.allclose(q8_Z, Q[8].Z))
        self.assertTrue(torch.allclose(qd8_Z, Qd[8].Z))
        self.assertTrue(torch.allclose(q21_Z, Q[21].Z))
        self.assertTrue(torch.allclose(qd21_Z, Qd[21].Z))
        self.assertTrue(torch.allclose(q79_Z, Q[79].Z))
        self.assertTrue(torch.allclose(qd79_Z, Qd[79].Z))


# Test cases to just make sure the trajectory is likely valid and matches expectation
# for the bernstein ARMOUR formulation
class TestBernsteinArmTrajectories(unittest.TestCase):
    def setUp(self):
        # initial conditions
        self.q = np.array([0, 0, 0])
        self.dq = np.array([np.pi, np.pi/2, np.pi/4])
        tdiscretization = 0.01
        duration = 1.

        # Params
        self.params = np.array([
            zp.polyZonotope([[0],[1]],1,id=0),
            zp.polyZonotope([[0],[1]],1,id=1),
            zp.polyZonotope([[0],[1]],1,id=2)
            ])
        self.param_range = np.ones(3) * np.pi/36
        self.param_range = np.vstack([-self.param_range, self.param_range])

        # Time discretization
        i_s = np.arange(int(duration/tdiscretization))
        gens = torch.ones(100) * tdiscretization/2
        centers = torch.as_tensor(tdiscretization*i_s+tdiscretization/2, dtype=torch.get_default_dtype())
        z = torch.vstack([centers, gens]).unsqueeze(2).transpose(0,1)
        self.times = zp.batchPolyZonotope(z, 1, id=3)
    
    def test_construction3link(self):
        zp.trajectories.BernsteinArmTrajectory(self.q, self.dq, self.q, self.params, self.param_range, tbrake=0.5, tfinal=1)

    def test_generation3link(self):
        gen = zp.trajectories.BernsteinArmTrajectory(self.q, self.dq, self.q, self.params, self.param_range, tbrake=0.5, tfinal=1)
        out = gen.getReference(np.array([self.times[i] for i in range(self.times.batch_shape[0])]))
        
        # Check outputs
        self.assertEqual(len(out), 3, "Expected tuple of 3 on output from getReference")
        self.assertEqual(len(out[0].T), 3, "Expected three joints on output from getReference")
        self.assertEqual(len(out[0].T[0]), 100, "Expected an output size of 100 for position output")
        self.assertEqual(len(out[1].T[0]), 100, "Expected an output size of 100 for velocity output")
        self.assertEqual(len(out[2].T[0]), 100, "Expected an output size of 100 for acceleration output")

        # Check ID's
        self.assertTrue(np.all(np.sort(out[1][0][0].id) == np.array([0,3])), "Expected id 0 and 3 for joint 0")
        self.assertTrue(np.all(np.sort(out[1][0][1].id) == np.array([1,3])), "Expected id 1 and 3 for joint 1")
        self.assertTrue(np.all(np.sort(out[1][0][2].id) == np.array([2,3])), "Expected id 2 and 3 for joint 2")

        # Remove dependence and compress
        Q_all = rem_dep(out[0], [0, 1, 2])
        Qd_all = rem_dep(out[1], [0, 1, 2])
        Qdd_all = rem_dep(out[2], [0, 1, 2])

        ## For the first link
        Q = Q_all.T[0]
        Qd = Qd_all.T[0]
        Qdd = Qdd_all.T[0]
        # Test 3 random values
        q8_Z = torch.tensor(
            [[0.256729518584417],
            [0.000469917940685359],
            [0.0141407622086939]]
        )
        qd8_Z = torch.tensor(
            [[2.79230721131860],
            [0.0158361157045228],
            [0.0403100714987681]]
        )
        qdd8_Z = torch.tensor(
            [[-7.55002469075804],
            [0.338000026608909],
            [0.350212968049739]]
        )

        q21_Z = torch.tensor(
            [[0.537481109675906],
            [0.00611641027723924],
            [0.00766274068657192]]
        )
        qd21_Z = torch.tensor(
            [[1.42605291361546],
            [0.0745735572508339],
            [0.0638873643898013]]
        )
        qdd21_Z = torch.tensor(
            [[-12.2481351325322],
            [0.503711185104137],
            [0.0528402794364413]]
        )

        q79_Z = torch.tensor(
            [[0.0728345191766624],
            [0.0818705988101653],
            [0.00497757671603243]]
        )
        qd79_Z = torch.tensor(
            [[-0.909704730967125],
            [0.0695361443304965],
            [0.0325903536808262]]
        )
        qdd79_Z = torch.tensor(
            [[5.99041243380992],
            [-0.503467711673483],
            [0.0497039227706078]]
        )
        self.assertTrue(torch.allclose(q8_Z, Q[8].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(qd8_Z, Qd[8].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(qdd8_Z, Qdd[8].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(q21_Z, Q[21].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(qd21_Z, Qd[21].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(qdd21_Z, Qdd[21].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(q79_Z, Q[79].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(qd79_Z, Qd[79].reduce_indep(1).Z))
        self.assertTrue(torch.allclose(qdd79_Z, Qdd[79].reduce_indep(1).Z))

    def test_generation3link_batch(self):
        gen = zp.trajectories.BernsteinArmTrajectory(self.q, self.dq, self.q, self.params, self.param_range)
        out = gen.getReference(self.times)
        
        # Check outputs
        self.assertEqual(len(out), 3, "Expected tuple of 3 on output from getReference")
        self.assertEqual(len(out[0]), 3, "Expected three joints on output from getReference")
        self.assertEqual(out[0][0].batch_shape[0], 100, "Expected a batch size of 100 for position output")
        self.assertEqual(out[1][0].batch_shape[0], 100, "Expected a batch size of 100 for velocity output")
        self.assertEqual(out[2][0].batch_shape[0], 100, "Expected a batch size of 100 for acceleration output")

        # Check ID's
        self.assertTrue(np.all(np.sort(out[1][0].id) == np.array([0,3])), "Expected id 0 and 3 for joint 0")
        self.assertTrue(np.all(np.sort(out[1][1].id) == np.array([1,3])), "Expected id 1 and 3 for joint 1")
        self.assertTrue(np.all(np.sort(out[1][2].id) == np.array([2,3])), "Expected id 2 and 3 for joint 2")

        # Remove dependence and compress
        Q_all = rem_dep(out[0], [0, 1, 2])
        Qd_all = rem_dep(out[1], [0, 1, 2])
        Qdd_all = rem_dep(out[2], [0, 1, 2])

        ## For the first link
        Q = Q_all[0].reduce_indep(1)
        Qd = Qd_all[0].reduce_indep(1)
        Qdd = Qdd_all[0].reduce_indep(1)
        # Test 3 random values
        q8_Z = torch.tensor(
            [[0.256729518584417],
            [0.000469917940685359],
            [0.0141407622086939]]
        )
        qd8_Z = torch.tensor(
            [[2.79230721131860],
            [0.0158361157045228],
            [0.0403100714987681]]
        )
        qdd8_Z = torch.tensor(
            [[-7.55002469075804],
            [0.338000026608909],
            [0.350212968049739]]
        )

        q21_Z = torch.tensor(
            [[0.537481109675906],
            [0.00611641027723924],
            [0.00766274068657192]]
        )
        qd21_Z = torch.tensor(
            [[1.42605291361546],
            [0.0745735572508339],
            [0.0638873643898013]]
        )
        qdd21_Z = torch.tensor(
            [[-12.2481351325322],
            [0.503711185104137],
            [0.0528402794364413]]
        )

        q79_Z = torch.tensor(
            [[0.0728345191766624],
            [0.0818705988101653],
            [0.00497757671603243]]
        )
        qd79_Z = torch.tensor(
            [[-0.909704730967125],
            [0.0695361443304965],
            [0.0325903536808262]]
        )
        qdd79_Z = torch.tensor(
            [[5.99041243380992],
            [-0.503467711673483],
            [0.0497039227706078]]
        )
        self.assertTrue(torch.allclose(q8_Z, Q[8].Z))
        self.assertTrue(torch.allclose(qd8_Z, Qd[8].Z))
        self.assertTrue(torch.allclose(qdd8_Z, Qdd[8].Z))
        self.assertTrue(torch.allclose(q21_Z, Q[21].Z))
        self.assertTrue(torch.allclose(qd21_Z, Qd[21].Z))
        self.assertTrue(torch.allclose(qdd21_Z, Qdd[21].Z))
        self.assertTrue(torch.allclose(q79_Z, Q[79].Z))
        self.assertTrue(torch.allclose(qd79_Z, Qd[79].Z))
        self.assertTrue(torch.allclose(qdd79_Z, Qdd[79].Z))



if __name__ == '__main__':
    unittest.main()


# # BERNSTEIN
# param_range_bs = np.ones(1) * np.pi/36
# param_range_bs = np.vstack([-param_range_bs, param_range_bs])


# # Discretize time


# # Make the trajectory generators. Test piecewise first
# gen_pw = zp.trajectories.PiecewiseArmTrajectory(q, dq, q, params, krange=self.param_range)
# Q, Qd, Qdd = gen_pw.getReference(times)

# # Remove dependence

# Q = rem_dep(Q, [0, 1, 2])
# Qd = rem_dep(Qd, [0, 1, 2])
# Qdd = rem_dep(Qdd, [0, 1, 2])

# # Check random values

# # Get the piecewise and
# from zonopy.joint_reachable_set.utils import remove_dependence_and_compress
# Q = remove_dependence_and_compress(Q[0][0], np.array(0))
# import zonopy.transformations.rotation as rot
# Q = rot.cos_sin_cartProd(Q, 1)

#         q0_Z = torch.tensor(
#             [[0.999876632438824, 0.0157073173104729],
#             [-2.05608302810082e-07, 1.30883545129187e-05],
#             [0.000371540638269134, 0],
#             [0, 0.0157491980017029]]
#             )
# gen_bs = zp.trajectories.BernsteinArmTrajectory(q, dq, q, params, krange=self.param_range)