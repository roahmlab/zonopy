'''
Define interval
Author: Qingyi Chen, Yongseok Kwon
Reference: CORA
'''

import torch

EMPTY_TENSOR = torch.tensor([])
class interval:
    '''
    I: <interval>

    inf: <torch.Tensor> infimum
    , shape [n,m]
    sup: <torch.Tensor> supremum
    , shape [n,m]

    Eq.
    I = { inf*(1-a)/2 + sup*(1+a)/2 | coef. a \in [-1,1] }
    '''
    def __init__(self, inf=EMPTY_TENSOR, sup=EMPTY_TENSOR, dtype=torch.get_default_dtype(), device=None):
        if isinstance(inf,list):
            inf = torch.tensor(inf)
        if isinstance(sup,list):
            sup = torch.tensor(sup)
        if inf.numel()==0 and sup.numel()!=0:
            inf = sup
        if inf.numel()!=0 and sup.numel()==0:
            sup = inf
        assert isinstance(inf, torch.Tensor) and isinstance(sup, torch.Tensor), "The inputs should be either torch tensor or list."
        assert inf.shape == sup.shape, "inf and sup is expected to be of the same shape"
        assert torch.all(inf <= sup), "inf should be less than sup entry-wise"

        self.__inf = inf.to(dtype=dtype, device=device)
        self.__sup = sup.to(dtype=dtype, device=device)
    @property
    def dtype(self):
        '''
        The data type of an interval properties
        return torch.float or torch.double        
        '''
        return self.inf.dtype
    @property
    def device(self):
        '''
        The device of an interval properties
        return 'cpu', 'cuda:0', or ...
        '''
        return self.inf.device
    @property
    def inf(self):
        '''
        The infimum of an interval
        return <torch.Tensor>
        ,shape [n,m]
        '''
        return self.__inf
    @inf.setter
    def inf(self,value):
        '''
        Set value of the infimum of an interval
        '''
        assert self.__inf.shape == value.shape
        self.__inf = value
    @property
    def sup(self):
        '''
        The supremum of an interval
        return <torch.Tensor>
        ,shape [n,m]
        '''
        return self.__sup
    @sup.setter
    def sup(self,value):
        '''
        Set value of the supremum of an interval
        '''
        assert self.__inf.shape == value.shape
        self.__inf = value
    @property
    def shape(self):
        '''
        The shape of elements (infimum or supremum) of an interval
        '''
        return tuple(self.__inf.shape)

    def to(self,dtype=None,device=None): 
        '''
        Change the device and data type of an interval
        dtype: torch.float or torch.double
        device: 'cpu', 'gpu', 'cuda:0', ...
        '''
        inf = self.__inf.to(dtype=dtype, device=device)
        sup = self.__sup.to(dtype=dtype, device=device)
        return interval(inf,sup)
    def cpu(self):    
        '''
        Change the device of an interval to CPU
        '''
        inf = self.__inf.cpu()
        sup = self.__sup.cpu()
        return interval(inf,sup)

    def __repr__(self):
        '''
        Representation of an interval as a text
        return <str>, 
        ex. interval(
               inf([0., 0.]),
               sup([1., 1.])
               )
        '''
        intv_repr1 = f"interval(\n"+str(self.__inf)+"," 
        intv_repr2 = "\n"+str(self.__sup) 
        intv_repr = intv_repr1.replace('tensor(','   inf(') + intv_repr2.replace('tensor(','   sup(')
        intv_repr = intv_repr.replace('    ','    ')
        return intv_repr+"\n   )"
    def __add__(self, other):
        '''
        Overloaded '+' operator for addition or Minkowski sum
        self: <interval>
        other: <torch.Tensor> OR <interval>
        return <interval>
        '''   
        if isinstance(other, interval):
            inf, sup = self.__inf+other.__inf, self.__sup+other.__sup
        elif isinstance(other, torch.Tensor) or isinstance(other, (int,float)):
            inf, sup = self.__inf+other, self.__sup+other
        else:
            assert False, f'the other object should be interval or numberic, but {type(other)}.'
        return interval(inf,sup)

    __radd__ = __add__ # '+' operator is commutative.

    def __sub__(self,other):
        '''
        Overloaded '-' operator for substraction or Minkowski difference
        self: <interval>
        other: <torch.Tensor> OR <interval>
        return <interval>
        '''   
        return self.__add__(-other)
    def __rsub__(self,other):
        '''
        Overloaded reverted '-' operator for substraction or Minkowski difference
        self: <interval>
        other: <torch.Tensor> OR <interval>
        return <interval>
        '''   
        return -self.__sub__(other)
    def __iadd__(self,other): 
        '''
        Overloaded '+=' operator for addition or Minkowski sum
        self: <interval>
        other: <torch.Tensor> OR <interval>
        return <interval>
        '''   
        return self+other
    def __isub__(self,other):
        '''
        Overloaded '-=' operator for substraction or Minkowski difference
        self: <interval>
        other: <torch.Tensor> OR <interval>
        return <interval>
        '''   
        return self-other
    def __pos__(self):
        '''
        Overloaded unary '+' operator for an interval ifself
        self: <interval>
        return <interval>
        '''   
        return self    
    def __neg__(self):
        '''
        Overloaded unary '-' operator for negation of an interval
        self: <interval>
        return <interval>
        '''   
        return interval(-self.__sup,-self.__inf)

    def __mul__(self, other):
        if isinstance(other,(int,float)):
            if other >= 0:
                return interval(other * self.__inf, other * self.__sup)
            else:
                return interval(other * self.__sup, other * self.__inf)

        if self.numel() == 1 and isinstance(other, interval):
            candidates = other.inf.repeat(4,1).reshape((4,) + other.shape)
            candidates[0] = self.__inf * other.__inf
            candidates[1] = self.__inf * other.__sup
            candidates[2] = self.__sup * other.__inf
            candidates[3] = self.__sup * other.__sup

            new_inf = torch.min(candidates,dim=0).values
            new_sup = torch.max(candidates,dim=0).values
            return interval(new_inf, new_sup)

        elif isinstance(other, interval) and (other.numel() == 1 or self.numel() == other.numel()):
            candidates = self.inf.repeat(4,1).reshape((4,) + self.shape)
            candidates[0] = self.__inf * other.__inf
            candidates[1] = self.__inf * other.__sup
            candidates[2] = self.__sup * other.__inf
            candidates[3] = self.__sup * other.__sup

            new_inf = torch.min(candidates,dim=0).values
            new_sup = torch.max(candidates,dim=0).values
            return interval(new_inf, new_sup)

        else:
            assert False, "such multiplication is not implemented yet"

    __rmul__ = __mul__

    def __getitem__(self, pos):
        inf = self.__inf[pos]
        sup = self.__sup[pos]
        return interval(inf, sup)

    def __setitem__(self, pos, value):
        # set one interval
        if isinstance(value, interval):
            self.__inf[pos] = value.__inf
            self.__sup[pos] = value.__sup
        else:
            self.__inf[pos] = value
            self.__sup[pos] = value

    def __len__(self):
        return len(self.__inf)

    def dim(self):
        return self.__inf.dim()

    def t(self):
        return interval(self.__inf.t(), self.__sup.t())

    def numel(self):
        return self.__inf.numel()
    def center(self):
        return (self.inf+self.sup)/2
    def rad(self):
        return (self.sup-self.inf)/2


#if __name__ == '__main__':

