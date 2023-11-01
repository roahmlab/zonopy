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
    '''
    matZono: <matZonotope>, <torch.float64>

    Z: <torch.Tensor> center vector and generator matrix Z = [c,G]
    , shape [N+1, nx, ny]
    center: <torch.Tensor> center matrix
    , shape [nx,ny] 
    generators: <torch.Tensor> generator tensor
    , shape [N, nx, ny]
    
    
    Eq. (coeff. a1,a2,...,aN \in [0,1])
    G = [[G1],[G2],...,[GN]]
    zono = C + a1*G1 + a2*G2 + ... + aN*GN
    '''
    def __init__(self,Z, dtype=None, device=None):
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