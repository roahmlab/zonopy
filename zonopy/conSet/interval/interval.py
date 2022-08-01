'''
Define interval
Author: Qingyi Chen, Yongseok Kwon
Reference: CORA
'''

import torch

EMPTY_TENSOR = torch.tensor([])
class interval:
    def __init__(self, inf=EMPTY_TENSOR, sup=EMPTY_TENSOR):
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

        self.__inf = inf
        self.__sup = sup
    @property
    def dtype(self):
        return self.inf.dtype
    @property
    def device(self):
        return self.inf.device
    @property
    def inf(self):
        return self.__inf
    @inf.setter
    def inf(self,value):
        assert self.__inf.shape == value.shape
        self.__inf = value
    @property
    def sup(self):
        return self.__sup
    @sup.setter
    def sup(self,value):
        assert self.__inf.shape == value.shape
        self.__inf = value
    @property
    def shape(self):
        return tuple(self.__inf.shape)

    def to(self,dtype=None,device=None):    
        inf = self.__inf.to(dtype=dtype, device=device)
        sup = self.__sup.to(dtype=dtype, device=device)
        return interval(inf,sup)
    def cpu(self):    
        inf = self.__inf.cpu()
        sup = self.__sup.cpu()
        return interval(inf,sup)
    '''
    def __str__(self):
        intv_str = f"interval of shape {self.shape}.\n inf: {self.__inf}.\n sup: {self.__sup}.\n"
        del_dict = {'tensor(':'','    ':' ',')':''}
        for del_el in del_dict.keys():
            intv_str = intv_str.replace(del_el,del_dict[del_el])

        return intv_str
    '''
    def __repr__(self):
        intv_repr1 = f"interval(\n"+str(self.__inf)+"," 
        intv_repr2 = "\n"+str(self.__sup) 
        intv_repr = intv_repr1.replace('tensor(','   inf(') + intv_repr2.replace('tensor(','   sup(')
        intv_repr = intv_repr.replace('    ','    ')
        return intv_repr+"\n   )"
    def __add__(self, other):
        if isinstance(other, interval):
            inf, sup = self.__inf+other.__inf, self.__sup+other.__sup
        elif isinstance(other, torch.Tensor) or isinstance(other, (int,float)):
            inf, sup = self.__inf+other, self.__sup+other
        else:
            assert False, f'the other object should be interval or numberic, but {type(other)}.'
        return interval(inf,sup)
    __radd__ = __add__
    def __sub__(self,other):
        return self.__add__(-other)
    def __rsub__(self,other):
        return -self.__sub__(other)
    def __iadd__(self,other): 
        return self+other
    def __isub__(self,other):
        return self-other
    def __pos__(self):
        return self    
    def __neg__(self):
        '''
        Overloaded unary '-' operator for negation
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

        elif isinstance(other, interval) and other.numel() == 1:
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

