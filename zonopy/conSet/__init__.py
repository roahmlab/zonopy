import torch
DEFAULT_DTYPE = torch.float
DEFAULT_ITYPE = torch.int
DEFAULT_DEVICE = 'cpu'

from zonopy.conSet.zonotope.zono import zonotope
from zonopy.conSet.zonotope.mat_zono import matZonotope
from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope
from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope
from zonopy.conSet.interval.interval import interval
