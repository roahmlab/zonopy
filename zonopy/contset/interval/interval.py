'''
Define interval
Author: Qingyi Chen, Yongseok Kwon
Reference: CORA
'''

import torch
import zonopy.internal as zpi

class interval:
    r""" N-rank tensor intervals

    An interval is a set of real numbers that includes all numbers between two given numbers.
    Here, we define an interval as a set of real numbers given infinum and supremum tensors
    :math:`\underbar{X}` and :math:`\overline{\text{X}}` such that
    :math:`\underbar{X} \leq X \leq \overline{\text{X}}`.
    
    .. math::

        \mathcal{I} := \left\{
            x \in \mathbb{R}^{n,m,...} 
            \; \middle\vert \;
            \begin{array}{c}
                \underbar{x}_{i,j,\ldots} \leq x_{i,j,\ldots} \leq \overline{\text{x}}_{i,j,\ldots} \\
                \forall{i=1,\ldots,n} \\
                \forall{j=1,\ldots,m} \\
                \vdots
            \end{array}
            \right\}
    
    """
    def __init__(self, inf=None, sup=None, dtype=None, device=None):
        """ Create an interval

        If ``inf`` and ``sup`` are both ``None``, an empty interval is created.
        If only one of ``inf`` or ``sup`` is ``None``, the interval is created as a point interval where ``inf = sup``.

        Args:
            inf (torch.Tensor, optional): infimum of the interval. Defaults to None.
            sup (torch.Tensor, optional): supremum of the interval. Defaults to None.
            dtype (torch.dtype, optional): data type of the interval. If None, the data type is inferred from the input tensors. Defaults to None.
            device (torch.device, optional): device of the interval. If None, the device is inferred from the input tensors. Defaults to None.
        
        Raises:
            AssertionError: If the shapes of ``inf`` and ``sup`` do not match.
            AssertionError: If the devices of ``inf`` and ``sup`` do not match.
            AssertionError: If ``inf`` is not less than or equal to ``sup`` entry-wise and :const:`zonopy.internal.__debug_extra__` is True.
        """
        if inf is None and sup is None:
            inf = torch.empty(0, dtype=dtype, device=device)
            sup = torch.empty(0, dtype=dtype, device=device)
        elif inf is None:
            inf = sup
        elif sup is None:
            sup = inf
        
        # Make sure that the input is a tensor
        inf = torch.as_tensor(inf)
        sup = torch.as_tensor(sup)
        
        # Promote the data type if necessary
        if dtype is None:
            dtype = torch.promote_types(inf.dtype, sup.dtype)
        
        inf = inf.to(dtype=dtype, device=device)
        sup = sup.to(dtype=dtype, device=device)
        
        assert inf.shape == sup.shape, "inf and sup are expected to be of the same shape"
        assert inf.device == sup.device, "inf and sup are expected to be on the same device"
        if zpi.__debug_extra__: assert torch.all(inf <= sup), "inf should be less than sup entry-wise"

        self.__inf = inf
        self.__sup = sup

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
        inf = self.__inf.to(dtype=dtype, device=device, non_blocking=True)
        sup = self.__sup.to(dtype=dtype, device=device, non_blocking=True)
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
    
    def __abs__(self):
        '''
        Overloaded unary operator for absolute value of an interval
        self: <interval>
        return <interval>
        '''   
        inf = torch.max(torch.stack([torch.zeros_like(self.__inf), self.__inf, -self.__sup]), dim=0).values
        sup = torch.max(torch.stack([self.__sup, -self.__inf, torch.zeros_like(self.__sup)]), dim=0).values
        return interval(inf,sup)

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
            # candidates = other.inf.repeat(4,1).reshape((4,) + other.shape)
            candidates = torch.empty((4,) + other.shape, dtype=other.inf.dtype, device=other.inf.device)
            candidates[0] = self.__inf * other.__inf
            candidates[1] = self.__inf * other.__sup
            candidates[2] = self.__sup * other.__inf
            candidates[3] = self.__sup * other.__sup

            new_inf = torch.min(candidates,dim=0).values
            new_sup = torch.max(candidates,dim=0).values
            return interval(new_inf, new_sup)

        elif isinstance(other, interval) and (other.numel() == 1 or self.numel() == other.numel()):
            # candidates = self.inf.repeat(4,1).reshape((4,) + self.shape)
            candidates = torch.empty((4,) + self.shape, dtype=self.inf.dtype, device=self.inf.device)
            candidates[0] = self.__inf * other.__inf
            candidates[1] = self.__inf * other.__sup
            candidates[2] = self.__sup * other.__inf
            candidates[3] = self.__sup * other.__sup

            new_inf = torch.min(candidates,dim=0).values
            new_sup = torch.max(candidates,dim=0).values
            return interval(new_inf, new_sup)

        elif isinstance(other, torch.Tensor):
            candidates = torch.empty((2,) + self.shape, dtype=self.inf.dtype, device=self.inf.device)
            candidates[0] = self.__inf * other
            candidates[1] = self.__sup * other

            new_inf = torch.min(candidates,dim=0).values
            new_sup = torch.max(candidates,dim=0).values
            return interval(new_inf, new_sup)

        else:
            assert False, "such multiplication is not implemented yet"

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other,(int,float)):
            if other > 0:
                return interval(self.__inf / other, self.__sup / other)
            else:
                return interval(self.__sup / other, self.__inf / other)

        elif isinstance(other, torch.Tensor):
            candidates = torch.empty((2,) + self.shape, dtype=self.inf.dtype, device=self.inf.device)
            candidates[0] = self.__inf / other
            candidates[1] = self.__sup / other

            new_inf = torch.min(candidates,dim=0).values
            new_sup = torch.max(candidates,dim=0).values
            return interval(new_inf, new_sup)

        elif isinstance(other, interval):
            y = 1 / other
            return self * y
        
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other,(int,float,torch.Tensor)):
            if isinstance(other, torch.Tensor) and len(other.shape) > 0:
                assert other.shape == self.shape, "other should be a scalar or have the same shape as self"
            else:
                other = torch.as_tensor(other, dtype=self.inf.dtype, device=self.inf.device).expand(self.shape)
            new_inf = torch.empty_like(self.__inf)
            new_sup = torch.empty_like(self.__sup)

            # if sup is 0, then inf is -inf or nan
            neg_infty_inf = self.__sup == 0
            # if both inf and sup are 0, then make nans
            make_nans = torch.logical_and(self.__inf == 0, neg_infty_inf)
            # we have intervals that cross zero, then make infinities
            zero_crossings = torch.logical_and(self.__inf < 0, self.__sup > 0)
            # rest of the intervals are safe
            safe_mask = torch.logical_not(zero_crossings | neg_infty_inf | make_nans)


            # Special cases
            new_inf[zero_crossings | neg_infty_inf] = -torch.inf
            new_sup[zero_crossings] = torch.inf
            new_sup[neg_infty_inf] = other[neg_infty_inf] / self.__inf[neg_infty_inf]
            new_inf[make_nans] = torch.nan
            new_sup[make_nans] = torch.nan
            
            # Compute the candidates for the regular case
            candidates = torch.empty((2,) + self.__inf[safe_mask].shape, dtype=self.inf.dtype, device=self.inf.device)
            candidates[0] = other[safe_mask] / self.__inf[safe_mask]
            candidates[1] = other[safe_mask] / self.__sup[safe_mask]

            new_inf[safe_mask] = torch.min(candidates,dim=0).values
            new_sup[safe_mask] = torch.max(candidates,dim=0).values
            return interval(new_inf, new_sup)
        # Flip division if not here
        return NotImplemented

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

    def __len__(self) -> int:
        """ Returns the length of the interval

        Returns:
            int: length of the interval (same as the first tensor dimension)
        """
        return len(self.__inf)

    def dim(self) -> int:
        """ Returns the number of dimensions of the interval
        
        Returns:
            int: number of dimensions of the interval
        """
        return self.__inf.dim()

    def t(self):
        """ Transposes the interval
        
        Returns:
            interval: transposed interval
        """
        return interval(self.__inf.t(), self.__sup.t())

    def numel(self) -> int:
        """ Returns the total number of elements in the interval

        Returns:
            int: number of elements in the interval
        """
        return self.__inf.numel()
    
    def center(self) -> torch.Tensor:
        """ Compute the center of the interval

        The center of the interval is the midpoint of the infimum and supremum.

        Returns:
            torch.Tensor: center of the interval
        """
        return (self.inf+self.sup)/2
    
    def rad(self) -> torch.Tensor:
        """ Compute the radius of the interval
        
        The radius of the interval is half of the difference between the supremum and infimum.
        It can be viewed as the distance from the center to the infimum or supremum.

        Returns:
            torch.Tensor: radius of the interval
        """
        return (self.sup-self.inf)/2


#if __name__ == '__main__':

