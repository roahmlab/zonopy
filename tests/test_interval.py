import unittest
import torch
from zonopy import interval

class TestInterval(unittest.TestCase):
    def setUp(self):
        self.inf = torch.tensor([1.0, 2.0])
        self.sup = torch.tensor([3.0, 4.0])
        self.I = interval(self.inf, self.sup)

    def test_addition(self):
        I2 = interval(torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0]))
        I3 = self.I + I2
        self.assertTrue(torch.allclose(I3.inf, torch.tensor([3.0, 5.0])))
        self.assertTrue(torch.allclose(I3.sup, torch.tensor([7.0, 9.0])))
        I4 = self.I + 1.0
        self.assertTrue(torch.allclose(I4.inf, torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.allclose(I4.sup, torch.tensor([4.0, 5.0])))
        I5 = self.I + torch.tensor([1.0, 2.0])
        self.assertTrue(torch.allclose(I5.inf, torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.allclose(I5.sup, torch.tensor([4.0, 6.0])))
        I6 = 1.2 + self.I
        self.assertTrue(torch.allclose(I6.inf, torch.tensor([2.2, 3.2])))
        self.assertTrue(torch.allclose(I6.sup, torch.tensor([4.2, 5.2])))
        I7 = torch.tensor([1.2, 2.2]) + self.I
        self.assertTrue(torch.allclose(I7.inf, torch.tensor([2.2, 4.2])))
        self.assertTrue(torch.allclose(I7.sup, torch.tensor([4.2, 6.2])))

    def test_subtraction(self):
        I2 = interval(torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0]))
        I3 = self.I - I2
        self.assertTrue(torch.allclose(I3.inf, torch.tensor([-3.0, -3.0])))
        self.assertTrue(torch.allclose(I3.sup, torch.tensor([1.0, 1.0])))
        I4 = self.I - 1.0
        self.assertTrue(torch.allclose(I4.inf, torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.allclose(I4.sup, torch.tensor([2.0, 3.0])))
        I5 = self.I - torch.tensor([1.0, 2.0])
        self.assertTrue(torch.allclose(I5.inf, torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.allclose(I5.sup, torch.tensor([2.0, 2.0])))
        I6 = 1.2 - self.I
        self.assertTrue(torch.allclose(I6.inf, torch.tensor([-1.8, -2.8])))
        self.assertTrue(torch.allclose(I6.sup, torch.tensor([0.2, -0.8])))
        I7 = torch.tensor([1.2, 2.2]) - self.I
        self.assertTrue(torch.allclose(I7.inf, torch.tensor([-1.8, -1.8])))
        self.assertTrue(torch.allclose(I7.sup, torch.tensor([0.2, 0.2])))

    def test_multiplication(self):
        I2 = interval(torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0]))
        I3 = self.I * I2
        self.assertTrue(torch.allclose(I3.inf, torch.tensor([2.0, 6.0])))
        self.assertTrue(torch.allclose(I3.sup, torch.tensor([12.0, 20.0])))
        I4 = self.I * -I2
        self.assertTrue(torch.allclose(I4.inf, torch.tensor([-12.0, -20.0])))
        self.assertTrue(torch.allclose(I4.sup, torch.tensor([-2.0, -6.0])))
        I5 = self.I * 3.0
        self.assertTrue(torch.allclose(I5.inf, torch.tensor([3.0, 6.0])))
        self.assertTrue(torch.allclose(I5.sup, torch.tensor([9.0, 12.0])))
        I6 = self.I * -3.0
        self.assertTrue(torch.allclose(I6.inf, torch.tensor([-9.0, -12.0])))
        self.assertTrue(torch.allclose(I6.sup, torch.tensor([-3.0, -6.0])))
        I7 = self.I * torch.tensor([3.0, 4.0])
        self.assertTrue(torch.allclose(I7.inf, torch.tensor([3.0, 8.0])))
        self.assertTrue(torch.allclose(I7.sup, torch.tensor([9.0, 16.0])))
        I8 = self.I * torch.tensor([-3.0, -4.0])
        self.assertTrue(torch.allclose(I8.inf, torch.tensor([-9.0, -16.0])))
        self.assertTrue(torch.allclose(I8.sup, torch.tensor([-3.0, -8.0])))
        I9 = 3.0 * self.I
        self.assertTrue(torch.allclose(I9.inf, torch.tensor([3.0, 6.0])))
        self.assertTrue(torch.allclose(I9.sup, torch.tensor([9.0, 12.0])))
        I10 = -3.0 * self.I
        self.assertTrue(torch.allclose(I10.inf, torch.tensor([-9.0, -12.0])))
        self.assertTrue(torch.allclose(I10.sup, torch.tensor([-3.0, -6.0])))
        I11 = torch.tensor([3.0, 4.0]) * self.I
        self.assertTrue(torch.allclose(I11.inf, torch.tensor([3.0, 8.0])))
        self.assertTrue(torch.allclose(I11.sup, torch.tensor([9.0, 16.0])))
        I12 = torch.tensor([-3.0, -4.0]) * self.I
        self.assertTrue(torch.allclose(I12.inf, torch.tensor([-9.0, -16.0])))
        self.assertTrue(torch.allclose(I12.sup, torch.tensor([-3.0, -8.0])))

    def test_center(self):
        c = self.I.center()
        self.assertTrue(torch.allclose(c, torch.tensor([2.0, 3.0])))

    def test_rad(self):
        r = self.I.rad()
        self.assertTrue(torch.allclose(r, torch.tensor([1.0, 1.0])))

if __name__ == '__main__':
    unittest.main()
