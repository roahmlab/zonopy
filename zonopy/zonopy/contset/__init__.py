"""
Continuous set representation
Author: Yongseok Kwon
"""
import torch as _torch

# TODO: REMOVE OR REINTRODUCE
class __DefaultOptions(object):
    __dtype: type = _torch.float
    __itype: type = _torch.int
    __device: str = 'cpu'
    def __repr__(self):
        return f'Default Options of Continuous Set\n - dtype: {self.__dtype}\n - itype: {self.__itype}\n - device: {self.__device}'
    def __str__(self):
        return self.__repr__()
    @property
    def DTYPE(self):
        return self.__dtype
    @property
    def ITYPE(self):
        return self.__itype
    @property
    def DEVICE(self):
        return self.__device
    def set(self, dtype=None,itype=None,device=None):
        if dtype is not None:
            if dtype == float:
                dtype = _torch.double
            assert dtype == _torch.float or dtype == _torch.double, 'Default dtype should be float.'
            self.__dtype = dtype
        if itype is not None:
            if itype == int:
                itype = _torch.int
            assert itype == _torch.int or itype == _torch.long or itype == _torch.short, 'Default itype should be integer.'
            self.__itype = itype
        if device is not None:
            self.__device = device

DEFAULT_OPTS = __DefaultOptions()

from .interval.interval import interval
from .zonotope.zono import zonotope
from .zonotope.mat_zono import matZonotope
from .zonotope.batch_zono import batchZonotope
from .zonotope.batch_mat_zono import batchMatZonotope
from .polynomial_zonotope.poly_zono import polyZonotope
from .polynomial_zonotope.mat_poly_zono import matPolyZonotope
from .polynomial_zonotope.batch_poly_zono import batchPolyZonotope
from .polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope

