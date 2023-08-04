"""
Continuous set representation
Author: Yongseok Kwon
"""
import torch 

class __DefaultOptions(object):
    __dtype: type = torch.float
    __itype: type = torch.int
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
                dtype = torch.double
            assert dtype == torch.float or dtype == torch.double, 'Default dtype should be float.'
            self.__dtype = dtype
        if itype is not None:
            if itype == int:
                itype = torch.int
            assert itype == torch.int or itype == torch.long or itype == torch.short, 'Default itype should be integer.'
            self.__itype = itype
        if device is not None:
            self.__device = device

DEFAULT_OPTS = __DefaultOptions()

