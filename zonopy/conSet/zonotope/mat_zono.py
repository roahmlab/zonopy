"""
Define class for matrix zonotope
Reference: CORA
Writer: Yongseok Kwon
"""
from zonopy.conSet.zonotope.zono import zonotope
from zonopy.conSet.utils import G_mul_c, G_mul_g, G_mul_C, G_mul_G
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
    def __init__(self,Z):

        assert type(Z) == torch.Tensor, f'The input matrix should be torch tensor, but {type(Z)}.'

        assert len(Z.shape) == 2 or len(Z.shape) == 3, f'The dimension of Z input should be either 2 or 3, but {len(Z.shape)}.'
        self.n_rows = Z.shape[0]
        self.n_cols = Z.shape[1]
        if len(Z.shape) == 2:
            Z = Z.reshape(self.n_rows,self.n_cols,1)
        
        Z = Z.to(dtype=torch.float32)
        self.Z = Z
        self.center = self.Z[:,:,0]
        self.generators = self.Z[:,:,1:]
        self.n_generators = self.Z.shape[-1] - 1

    def __matmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a __ with a matZonotope
        self: <matZonotope>
        other: <torch.tensor> OR <zonotope>
        return <zonotope>
        
        other: <matZonotope>
        return <matZonotope>
        '''
        if type(other) == torch.Tensor:
            assert len(other.shape) == 1, 'The other object should be 1-D tensor.'  
            assert other.shape[0] == self.n_cols
            z = G_mul_c(self.Z,other)    
            return zonotope(z)
    
        elif type(other) == zonotope:
            assert self.n_cols == other.dim
            z = G_mul_g(self.Z,other.Z)
            return zonotope(z)

        elif type(other) == matZonotope:
            assert self.n_cols == other.n_rows
            dims = [self.n_rows, self.n_cols, other.n_cols]
            Z = G_mul_G(self.Z,other.Z,dims)
            return matZonotope(Z)

        else:
            raise ValueError('the other object should be torch tensor or polynomial zonotope.')

    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a __ with a matZonotope
        self: <matZonotope>
        other: <torch.tensor> OR <zonotope>
        return <zonotope>
        
        other: <matZonotope>
        return <matZonotope>
        '''
        if type(other) == torch.Tensor:
            assert len(other.shape) == 2, 'The other object should be 2-D tensor.'  
            assert other.shape[1] == self.n_rows    
            Z = other @ self.Z
            return matZonotope(Z)

    def deleteZerosGenerators(self):
        '''
        delete zero vector generators
        self: <matZonotope>

        return <matZonotope>
        '''
        non_zero_idxs = torch.any(torch.any(self.generators!=0,axis=0),axis=0).to(dtype=int)
        Z_new = torch.zeros(self.n_rows,self.n_cols,sum(non_zero_idxs)+1)
        j=0
        for i in range(self.n_generators):
            if non_zero_idxs[i]:
                j += 1
                Z_new[:,:,j] = self.generators[:,:,i]
        Z_new[:,:,0] = self.center
        return matZonotope(Z_new)
