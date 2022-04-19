'''
Define interval
Author: Qingyi Chen, Yongseok Kwon
Reference: CORA
'''

import torch
from zonopy.conSet import DEFAULT_OPTS
from torch import Tensor

EMPTY_TENSOR = torch.tensor([])
class interval:
    def __init__(self, inf=EMPTY_TENSOR, sup=EMPTY_TENSOR,dtype=None,device=None):
        if dtype is None:
            dtype = DEFAULT_OPTS.DTYPE
        if device is None:
            device = DEFAULT_OPTS.DEVICE
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

        self.__inf = inf.to(dtype=dtype,device=device)
        self.__sup = sup.to(dtype=dtype,device=device)
        self.__shape = tuple(inf.shape)
        self.__dtype = dtype
        self.__device = device
    @property
    def inf(self):
        return self.__inf
    @inf.setter
    def inf(self,value):
        assert self.__inf.shape == value.shape
        self.__inf = value.to(dtype=self.__dtype,device=self.__device)
    @property
    def sup(self):
        return self.__sup
    @sup.setter
    def sup(self,value):
        assert self.__inf.shape == value.shape
        self.__inf = value.to(dtype=self.__dtype,device=self.__device)

    @property
    def shape(self):
        return self.__shape
    @property
    def dtype(self):
        return self.__dtype
    @property
    def device(self):
        return self.__device
    def __str__(self):
        intv_str = f"interval of shape {self.shape}.\n inf: {self.__inf}.\n sup: {self.__sup}.\n"
        del_dict = {'tensor(':'','    ':' ',')':''}
        for del_el in del_dict.keys():
            intv_str = intv_str.replace(del_el,del_dict[del_el])

        return intv_str
    def __repr__(self):
        intv_repr1 = f"interval(\n{self.__inf}," 
        intv_repr2 = f"\n{self.__sup}" 
        intv_repr = intv_repr1.replace('tensor(','   inf(') + intv_repr2.replace('tensor(','   sup(')
        intv_repr = intv_repr.replace('    ','    ')
        return intv_repr+")"
    def __add__(self, other):
        if isinstance(other, interval):
            inf, sup = self.__inf+other.__inf, self.__sup+other.__sup
        elif isinstance(other, torch.Tensor) or isinstance(other, (int,float)):
            inf, sup = self.__inf+other, self.__sup+other
        else:
            assert False, f'the other object should be interval or numberic, but {type(other)}.'
        return interval(inf,sup,self.__dtype,self.__device)
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
        return interval(-self.__sup,-self.__inf,self.__dtype,self.__device)

    def __mul__(self, other):
        if isinstance(other,(int,float)):
            if other >= 0:
                return interval(other * self.__inf, other * self.__sup, self.__dtype, self.__device)
            else:
                return interval(other * self.__sup, other * self.__inf, self.__dtype, self.__device)

        if self.numel() == 1 and isinstance(other, interval):
            candidates = other.inf.repeat(4,1).reshape((4,) + other.shape)
            candidates[0] = self.__inf * other.__inf
            candidates[1] = self.__inf * other.__sup
            candidates[2] = self.__sup * other.__inf
            candidates[3] = self.__sup * other.__sup

            new_inf = torch.min(candidates,dim=0).values
            new_sup = torch.max(candidates,dim=0).values
            return interval(new_inf, new_sup, self.__dtype, self.__device)

        elif isinstance(other, interval) and other.numel() == 1:
            candidates = self.inf.repeat(4,1).reshape((4,) + self.shape)
            candidates[0] = self.__inf * other.__inf
            candidates[1] = self.__inf * other.__sup
            candidates[2] = self.__sup * other.__inf
            candidates[3] = self.__sup * other.__sup

            new_inf = torch.min(candidates,dim=0).values
            new_sup = torch.max(candidates,dim=0).values
            return interval(new_inf, new_sup, self.__dtype, self.__device)

        else:
            assert False, "such multiplication is not implemented yet"

    __rmul__ = __mul__

    def __getitem__(self, pos):
        inf = self.__inf[pos]
        sup = self.__sup[pos]
        return interval(inf, sup, self.__dtype, self.__device)

    def __setitem__(self, pos, value):
        # set one interval
        if isinstance(value, interval):
            self.__inf[pos] = value.__inf.to(dtype = self.__dtype, device = self.__device)
            self.__sup[pos] = value.__sup.to(dtype = self.__dtype, device = self.__device)
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
    def to(self,dtype=None,device=None):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return interval(self.__inf, self.__sup, self.__dtype, self.__device)


    '''


    def __truediv__(self, other):
        inverse = None
        if other.inf == other.sup == 0:
            return interval(Tensor([]).to(self.dtype).to(self.device))
        elif other.inf < 0 < other.sup:
            inverse = interval(Tensor([math.inf, math.inf]).to(self.dtype).to(self.device))
        elif other.inf < 0 and other.sup == 0:
            inverse = interval(Tensor([-math.inf, 1 / other.inf]).to(self.dtype).to(self.device))
        elif other.inf == 0 and other.sup > 0:
            inverse = interval(Tensor([1 / other.sup, math.inf]).to(self.dtype).to(self.device))
        else:
            inverse = interval(Tensor([1 / other.sup, 1 / other.inf]).to(self.dtype).to(self.device))

        return self.__mul__(inverse)

    def __and__(self, other):
        if self.is_empty or other.is_empty:
            return interval(Tensor([]).to(self.dtype).to(self.device))

        new_inf = max(self.inf, other.inf)
        new_sup = min(self.sup, other.sup)
        if new_inf <= new_sup:
            return interval(Tensor([new_inf, new_sup]).to(self.dtype).to(self.device))
        else:
            return interval(Tensor([]).to(self.dtype).to(self.device))

    def __or__(self, other):
        if self.is_empty:
            return interval(other.interval_tensor)
        elif other.is_empty:
            return interval(self.interval_tensor)

        if max(self.inf, other.inf) <= min(self.sup, other.sup):
            new_inf = min(self.inf, other.inf)
            new_sup = max(self.sup, other.sup)
            return interval(Tensor([new_inf, new_sup]).to(self.dtype).to(self.device))
        else:
            return interval(Tensor([math.nan, math.nan]).to(self.dtype).to(self.device))

    def exp(self):
        return interval(torch.exp(self.interval_tensor))

    def log(self):
        if self.inf >=0:
            return interval(torch.log(self.interval_tensor))
        else:
            print("error in log")

    def abs(self):
        if self.sup < 0:
            return interval(torch.abs(Tensor([self.sup, self.inf]).to(self.dtype).to(self.device)))
        elif x.inf > 0:
            return interval(self.interval_tensor)
        else:
            return interval(Tensor([0, torch.max(torch.abs(self.interval_tensor))]))

    def sin(self):
        y_inf = self.inf % (2 * math.pi) / (math.pi / 2)
        y_sup = se;f.sup % (2 * math.pi) / (math.pi / 2)

        if (self.sup - self.inf >= 2 * math.pi) or (0 <= y_inf < 1 and 0 <= y_sup < 1 and y_inf > y_sup) or 
            (0 <= y_inf < 1 and 3 <= y_sup < 4) or (1 <= y_inf < 3 and 1<= y_sup < 3 and y_inf > y_sup):
            return interval(Tensor([-1, 1]).to(self.dtype).to(self.device))
        elif (0 <= y_inf < 1 and 0 <= y_sup < 1 and y_inf <= y_sup) or (3 <= y_inf < 4 and 0 <= y_sup < 1) or 
            (3 <= y_inf < 4 and 3 <= y_sup < 4 and y_inf <= y_sup):
            return interval(torch.sin(self.interval_tensor))
        elif (0 <= y_inf < 1 and 1 <= y_sup < 3) or (3 <= y_inf < 4 and 1 <= y_sup < 3):
            return interval(Tensor([torch.min(torch.sin(self.interval_tensor)),1]).to(self.dtype).to(self.device))
        elif (1 <= y_inf < 3 and 0 <= y_sup < 1) or (1 <= y_inf < 3 and 3 <= y_sup < 4):
            return interval(Tensor([-1,torch.max(torch.sin(self.interval_tensor))]).to(self.dtype).to(self.device))
        elif 1 <= y_inf < 3 and 1 <= y_sup < 3 and y_inf <= y_sup:
            return interval(torch.sin(Tensor([self.y_sup, self.y_inf]).to(self.dtype).to(self.device)))

    def cos(self):
        y_inf = self.inf % (2 * math.pi) / (math.pi / 2)
        y_sup = se;f.sup % (2 * math.pi) / (math.pi / 2)

        if (self.sup - self.inf >= 2 * math.pi) or (0 <= y_inf < 2 and 0 <= y_sup < 2 and y_inf > y_sup) or 
            (2 <= y_inf < 4 and 2 <= y_sup < 4 and y_inf > y_sup):
            return interval(Tensor([-1, 1]).to(self.dtype).to(self.device))
        elif (2 <= y_inf < 4 and 2 <= y_sup < 4 and y_inf <= y_sup):
            return interval(torch.cos(self.interval_tensor))
        elif (2 <= y_inf < 4 and 0 <= y_sup < 2):
            return interval(Tensor([torch.min(torch.cos(self.interval_tensor)),1]).to(self.dtype).to(self.device))
        elif (0 <= y_inf < 2 and 2 <= y_sup < 4):
            return interval(Tensor([-1,torch.max(torch.sin(self.interval_tensor))]).to(self.dtype).to(self.device))
        elif 1 <= y_inf <= 3 and 1 <= y_sup <= 3 and y_inf <= y_sup:
            return interval(torch.cos(Tensor([self.y_sup, self.y_inf]).to(self.dtype).to(self.device)))
    '''

def matmul_interval(mat, intv):
    assert isinstance(mat, Tensor) or isinstance(mat, interval), "the matrix should be in the type of a torch tensor or an interval"
    assert isinstance(intv, interval) or isinstance(mat, interval), "the intv or the mat should be in the type of an interval"
    assert mat.dim() == 2, "the dimension of the matrix should be 2"
    assert intv.dim() == 2, "the dimenstion of the interval matrix should be 2"
    assert mat.shape[1] == intv.shape[0], "the dimension of mat and interval should match"

    I, K = mat.shape
    K, J = intv.shape

    new_inf = torch.zeros(I,J).to(intv.dtype).to(intv.device)
    new_sup = torch.zeros_like(new_inf)

    for i in range(I):
        for k in range(K):
            for j in range(J):
                new_intv = mat[i,k] * intv[k,j]
                new_inf[i,j] += new_intv.inf.item()
                new_sup[i,j] += new_intv.sup.item()

    return interval(new_inf, new_sup)


def cross_interval(vec, intv):
    assert vec.numel() == 3, "we are considering only 3d vec"
    if isinstance(vec, Tensor):
        new_inf = torch.zeros_like(vec).to(intv.dtype).to(intv.device)
    elif isinstance(vec, interval):
        new_inf = torch.zeros_like(vec.inf).to(intv.dtype).to(intv.device)
    else:
        assert False, "such cross not supported"
    new_sup = torch.zeros_like(new_inf)

    intvs = []
    intvs.append(vec[1] * intv[2] - vec[2] * intv[1])
    intvs.append(vec[2] * intv[0] - vec[0] * intv[2])
    intvs.append(vec[0] * intv[1] - vec[1] * intv[0])

    for i in range(3):
        new_inf[i] = intvs[i].inf
        new_sup[i] = intvs[i].sup

    return interval(new_inf, new_sup)


if __name__ == '__main__':
    print("testing the functionality...")
    print("--- class construction test... ---")
    inf = torch.Tensor([1])
    sup = torch.Tensor([2])
    intv = interval(inf, sup)
    print(f"interval is: {intv}")

    print("--- testing index... ---")
    inf1 = torch.Tensor([[0.9,1.9],[2.9,3.9]])
    sup1 = torch.Tensor([[1.1,2.1],[3.1,4.1]])
    intv1 = interval(inf1, sup1)
    print(f"intv[0,0] = {intv1[0,0]}")
    print(f"intv[0] = {intv1[0]}")


    print("--- basic operation test... ---")
    intv1 = interval(torch.Tensor([-4]), torch.Tensor([2]))
    intv2 = interval(torch.Tensor([-12]), torch.Tensor([0]))
    print(f"intv1 + intv2 = {intv1 + intv2}")
    print(f"intv1 - intv2 = {intv1 - intv2}")
    print(f"intv1 * intv2 = {intv1 * intv2}")

    print("--- cross test... ---")
    intv1 = interval(torch.Tensor([0.9,1.9,2.9]), torch.Tensor([1.1,2.1,3.1]))
    intv2 = interval(torch.Tensor([0.9,4.9,-0.1]), torch.Tensor([1.1,5.1,0.1]))
    print(f"intv1 x intv2 = {cross_interval(intv1, intv2)}")
    

    print("--- matrix interval test... ---")
    inf1 = torch.Tensor([[0.9,1.9],[2.9,3.9]])
    sup1 = torch.Tensor([[1.1,2.1],[3.1,4.1]])
    intv1 = interval(inf1, sup1)

    inf2 = torch.Tensor([[9.9,9.9],[1.9,1.9]])
    sup2 = torch.Tensor([[10.1,10.1],[2.1,2.1]])
    intv2 = interval(inf2, sup2)
    print(f"intv1 matmul intev2 = {matmul_interval(intv1, intv2)}")
























