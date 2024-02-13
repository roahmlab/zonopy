"""
Define class for matrix zonotope
Author: Yongseok Kwon
Reference:
"""
from zonopy.contset.zonotope.zono import zonotope
from zonopy.contset.zonotope.utils import pickedGenerators
import torch
from ..gen_ops import (
    _matmul_genmzono_impl,
    )

class matZonotope():
    r""" 2D Matrix Zonotope class for representing a zonotope in matrix form.

    The matrix zonotope is defined as a set of linear combinations of a center vector and generator matrix.
    Similar to the :class:`zonotope` class, the matrix zonotope is defined as a set of linear combinations of
    a center vector and generator matrix.
    In this case, it is a set of the form:

    .. math::
        \mathcal{Z} := \left\{ C + \sum_{i=1}^{N} a_i G_i \; \middle\vert \; a_i \in [0,1] \right\}

    where :math:`C` is the center matrix and :math:`G_i` are the generator matrices.

    Here, we define :math:`\mathbf{Z}` as a tensor of shape :math:`(N+1) \times dx \times dy` where :math:`dx` and :math:`dy` are the
    number of rows and columns of all matrices, respectively.
    That is, :math:`\mathbf{Z} = [C, G_1, G_2, \ldots, G_N]`.
    """
    def __init__(self, Z, dtype=None, device=None):
        r''' Initialize the matrix zonotope with a center and generator matrix.
        
        Args:
            Z (torch.Tensor): The center and generator matrix of the matrix zonotope :math:`\mathbf{Z} = [C, G_1, G_2, \ldots, G_N]`
            dtype (torch.dtype, optional): The data type of the matrix zonotope. If ``None``, it will be inferred. Default: ``None``
            device (torch.device, optional): The device of the matrix zonotope. If ``None``, it will be inferred. Default: ``None``

        Raises:
            AssertionError: If the dimension of the input :math:`\mathbf{Z}` is not 3.
        '''
        # Make sure Z is a tensor
        if not isinstance(Z, torch.Tensor) and dtype is None:
            dtype = torch.get_default_dtype()
        Z = torch.as_tensor(Z, dtype=dtype, device=device)
        assert len(Z.shape) == 3, f'The dimension of Z input should be 3, not {len(Z.shape)}.'
        self.Z = Z
    @property
    def dtype(self):
        return self.Z.dtype
    @property
    def device(self):
        return self.Z.device   
    @property
    def center(self):
        return self.Z[0]
    @center.setter
    def center(self,value):
        self.Z[0] = value
    @property
    def generators(self):
        return self.Z[1:]
    @generators.setter
    def generators(self,value):
        self.Z[1:] = value
    @property 
    def shape(self):
        return tuple(self.Z.shape[1:])
    @property
    def n_rows(self):
        return self.Z.shape[1]
    @property
    def n_cols(self):
        return self.Z.shape[2]
    @property
    def n_generators(self):
        return len(self.Z)-1
    @property
    def T(self):
        return matZonotope(self.Z.transpose(1,2))

    def to(self,dtype=None,device=None):
        Z = self.Z.to(dtype=dtype,device=device, non_blocking=True)
        return matZonotope(Z)
    def cpu(self):
        Z = self.Z.cpu()
        return matZonotope(Z)

    def __matmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a __ with a matZonotope
        self: <matZonotope>
        other: <torch.tensor> OR <zonotope>
        return <zonotope>
        
        other: <matZonotope>
        return <matZonotope>
        '''
        if isinstance(other, torch.Tensor):
            assert len(other.shape) == 1, 'The other object should be 1-D tensor.'  
            assert other.shape[0] == self.n_cols
            z = self.Z@other
            return zonotope(z)
        
        elif isinstance(other,zonotope):
            # Shim to matZonotope
            shim_other = matZonotope(other.Z.unsqueeze(-1))
            Z = _matmul_genmzono_impl(self, shim_other)
            return zonotope(Z.squeeze(-1))
        
        elif isinstance(other,matZonotope):
            Z = _matmul_genmzono_impl(self, other)
            return matZonotope(Z)
        
        else:
            return NotImplemented

    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a __ with a matZonotope
        self: <matZonotope>
        other: <torch.tensor> OR <zonotope>
        return <zonotope>
        
        other: <matZonotope>
        return <matZonotope>
        '''
        if isinstance(other, torch.Tensor):
            assert len(other.shape) == 2, 'The other object should be 2-D tensor.'  
            assert other.shape[1] == self.n_rows 
            Z = other @ self.Z 
            return matZonotope(Z)
        else:
            assert False, 'Invalid object for reversed matrix multiplication with matrix zonotope.'


    def deleteZerosGenerators(self,eps=0):
        '''
        delete zero vector generators
        self: <matZonotope>

        return <matZonotope>
        '''
        non_zero_idxs = torch.any(torch.any(abs(self.generators)>eps,axis=1),axis=1)
        Z = torch.cat((self.center.unsqueeze(0),self.generators[non_zero_idxs]),0)
        return matZonotope(Z)

    def reduce(self,order,option='girard'):
        if option == 'girard':
            Z = self.deleteZerosGenerators()
            if order == 1:
                center, G = Z.center,Z.generators
                d = torch.sum(abs(G),0).reshape(-1)
                Gbox = torch.diag(d).reshape(-1,self.n_rows,self.n_cols)
                ZRed = torch.cat((center.reshape(-1,self.n_rows,self.n_cols),Gbox),0)
            else:
                center, Gunred, Gred = pickedGenerators(Z.center,Z.generators,order)
                d = torch.sum(abs(Gred),0).reshape(-1)
                Gbox = torch.diag(d).reshape(-1,self.n_rows,self.n_cols)
                ZRed = torch.cat((center.reshape(-1,self.n_rows,self.n_cols),Gunred,Gbox),0)
            return matZonotope(ZRed)
        else:
            assert False, 'Invalid reduction option'