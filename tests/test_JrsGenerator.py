import numpy as np
import torch
import zonopy as zp
from zonopy.conSet.polynomial_zonotope.utils import remove_dependence_and_compress
import zonopy.transformations.rotation as rot
import unittest

# Helper
rem_dep = np.vectorize(remove_dependence_and_compress, excluded=[1])

# Test cases using the bernstein trajectory (validates math)
# Values are calculated using reference implementation in MATLAB
class TestJrsGenerator(unittest.TestCase):
    def setUp(self):
        # initial conditions
        self.q = np.array([0, 0, 0])
        self.dq = np.array([np.pi, np.pi/2, np.pi/4])

    def testSinCos(self):
        tdiscretization = 0.01
        duration = 1.
        # Params
        params = np.array([zp.polyZonotope([[0],[1]],1,id=0)])
        param_range = np.ones(3) * np.pi/36
        param_range = np.vstack([-param_range, param_range])

        # Time discretization
        i_s = np.arange(int(duration/tdiscretization))
        gens = torch.ones(100) * tdiscretization/2
        centers = torch.as_tensor(tdiscretization*i_s+tdiscretization/2, dtype=torch.get_default_dtype())
        z = torch.vstack([centers, gens]).unsqueeze(2).transpose(0,1)
        times = zp.batchPolyZonotope(z, 1, id=3)
        gen = zp.trajectories.BernsteinArmTrajectory(self.q, self.dq, self.q, params, param_range, tbrake=0.5, tfinal=1)
        out = gen.getReference(times)
        Q_all = rem_dep(out[0], [0, 1, 2])

        # Make sure sincos is working right for later steps
        Q = rot.cos_sin_cartProd(Q_all[0],1)
        self.assertTrue(Q.id == 0, "Expected id 0 for joint 0")

        # Test 3 random values
        q5_Z = torch.tensor(
            [[0.985605608771582, 0.169060861840339],
            [-2.25653912247780e-05, 0.000131553665967734],
            [0.00265660838713821, 0],
            [0, 0.0148457361706722]]
        )
        q65_Z = torch.tensor(
            [[0.967660274436210, 0.247560651411644],
            [-0.0167143671096599, 0.0653122331543875],
            [0.00379916868626260, 0],
            [0, 0.00871093649576110]]
        )
        q77_Z = torch.tensor(
            [[0.994136249324561, 0.0917957646470413],
            [-0.00740147933612465, 0.0800376523912375],
            [0.00259895993594730, 0],
            [0, 0.00595835315446117]]
        )
        self.assertTrue(torch.allclose(q5_Z, Q[5].Z))
        self.assertTrue(torch.allclose(q65_Z, Q[65].Z))
        self.assertTrue(torch.allclose(q77_Z, Q[77].Z))


if __name__ == '__main__':
    unittest.main()