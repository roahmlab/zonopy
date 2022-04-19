"""
Define class for matrix zonotope
Author: Yongseok Kwon
Reference:
"""
from zonopy.conSet import DEFAULT_OPTS
from zonopy.conSet.zonotope.zono import zonotope
from zonopy.conSet.utils import G_mul_c, G_mul_g, G_mul_C, G_mul_G
from zonopy.conSet.zonotope.utils import pickedGenerators
import torch

EMPTY_TENSOR = torch.tensor([])
class matZonotope():
    '''
    matZono: <matZonotope>, <torch.float64>

    Z: <torch.Tensor> center vector and generator matrix Z = [c,G]
    , shape [nx, ny, N+1] OR [nx, ny], where N = 0
    -> shape [nx, N+1]
    center: <torch.Tensor> center matrix
    , shape [nx,ny, 1] 
    generators: <torch.Tensor> generator tensor
    , shape [nx, ny, N]
    
    
    Eq. (coeff. a1,a2,...,aN \in [0,1])
    G = [G1,G2,...,GN]
    zono = C + a1*G1 + a2*G2 + ... + aN*GN
    '''
    def __init__(self,Z=EMPTY_TENSOR,dtype=None,device=None):
        if dtype is None:
            dtype = DEFAULT_OPTS.DTYPE
        if device is None:
            device = DEFAULT_OPTS.DEVICE
        if isinstance(Z,list):
            Z = torch.tensor(Z)
        assert isinstance(Z,torch.Tensor), f'The input matrix should be torch tensor, but {type(Z)}.'

        assert len(Z.shape) == 2 or len(Z.shape) == 3, f'The dimension of Z input should be either 2 or 3, but {len(Z.shape)}.'
        if len(Z.shape) == 2:
            Z = Z.reshape(Z.shape[0],Z.shape[1],1)

        self.__dtype = dtype
        self.__device = device  
        self.Z = Z.to(dtype=dtype,device=device)
        self.__center = self.Z[:,:,0]
        self.__generators = self.Z[:,:,1:]
    @property
    def dtype(self):
        return self.__dtype
    @property
    def device(self):
        return self.__device    
    @property
    def center(self):
        return self.__center
    @center.setter
    def center(self,value):
        self.Z[:,:,0] = self.__center  = value.to(dtype=self.__dtype,device=self.__device)
    @property
    def generators(self):
        return self.__generators
    @generators.setter
    def generators(self,value):
        value = value.to(dtype=self.__dtype,device=self.__device)
        self.Z = torch.cat((self.__center.reshape(self.n_rows,self.n_cols,1),value),dim=-1)
        self.__generators = value
    @property
    def n_rows(self):
        return self.Z.shape[0]
    @property
    def n_cols(self):
        return self.Z.shape[1]
    @property
    def n_generators(self):
        return self.__generators.shape[-1]
    @property
    def T(self):
        return matZonotope(self.Z.permute(1,0,2),self.__dtype,self.__device)

    def to(self,dtype=None,device=None):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return matZonotope(self.Z,dtype,device)

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
            z = G_mul_c(self.Z,other)    
            return zonotope(z,self.device,self.dtype)
    
        elif isinstance(other,zonotope):
            assert self.n_cols == other.dimension
            z = G_mul_g(self.Z,other.Z)
            return zonotope(z,self.device,self.dtype)

        elif isinstance(other,matZonotope):
            assert self.n_cols == other.n_rows
            dims = [self.n_rows, self.n_cols, other.n_cols]
            Z = G_mul_G(self.Z,other.Z,dims)
            return matZonotope(Z,self.device,self.dtype)

        else:
            assert False, 'Invalid object for matrix multiplication with matrix zonotope.'

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
            return matZonotope(Z,self.device,self.dtype)
        else:
            assert False, 'Invalid object for reversed matrix multiplication with matrix zonotope.'


    def deleteZerosGenerators(self,eps=0):
        '''
        delete zero vector generators
        self: <matZonotope>

        return <matZonotope>
        '''
        non_zero_idxs = torch.any(torch.any(abs(self.generators)>eps,axis=0),axis=0)
        Z_new = torch.zeros(self.n_rows,self.n_cols,sum(non_zero_idxs)+1)
        Z_new[:,:,0] = self.center
        Z_new[:,:,1:] = self.generators[:,:,non_zero_idxs]
        return matZonotope(Z_new,self.__dtype,self.__device)

    def reduce(self,order,option='girard'):
        if option == 'girard':
            Z = self.deleteZerosGenerators()
            center, Gunred, Gred = pickedGenerators(Z.center,Z.generators,order)
            d = torch.sum(abs(Gred),-1).reshape(-1)
            Gbox = torch.diag(d).reshape(self.n_rows,self.n_cols,-1)
            ZRed = torch.cat((center.reshape(self.n_rows,self.n_cols,-1),Gunred,Gbox),dim=-1)
            return matZonotope(ZRed,self.__dtype,self.__device)
        else:
            assert False, 'Invalid reduction option'
        return
