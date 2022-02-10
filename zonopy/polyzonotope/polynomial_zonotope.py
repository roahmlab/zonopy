"""
Reference: CORA
"""
from utils import removeRedundantExponents
import torch
import numpy as np

EMPTY_TENSOR = torch.tensor([])
class polyZonotope:
    '''
    pZ: <polyZonotope>
    
    c: <torch.Tensor> center of the polyonmial zonotope
    , shape: [nx,1] 
    G: <torch.Tensor> generator matrix containing the dependent generators
    , shape: [nx, N]
    Grest: <torch.Tensor> generator matrix containing the independent generators
    , shape: [nx, M]
    expMat: <troch.Tensor> matrix containing the exponents for the dependent generators
    , shape: [p, N]
    id: <torch.Tensor> vector containing the integer identifiers for the dependent factors
    , shape: [p, 1]


    # NOTE: need to be confirmed
    Eq. (coeff. a1,a2,...,aN; b1,b2,...,bp ~ [0,1])
    G = [gd1,gd2,...,gdN]
    Grest = [gi1,gi2,...,giM]
    expMat = [[i11,i12,...,i1M],[i21,i22,...,i2M],...,[ip1,ip2,...,ipM]]
    id = [0,1,2,...,p-1]

    pZ = c + a1*gd1 + a2*gd2 + ... + aN*gdN + b1^i11*b2^i21*...*bp^ip1*gi1 + b1^i12*b2^i22*...*bp^ip2*gi2 + ... 
    + b1^i1M*b2^i2M*...*bp^ipM*giM
    
    '''
    def __init__(self,c=EMPTY_TENSOR,G=EMPTY_TENSOR,Grest=EMPTY_TENSOR,expMat=None,id=None,device='cpu'):
        
        # TODO: need to allow np.ndarray type
        # TODO: assign device

        if type(c) == np.ndarray:
            c = torch.from_numpy(c)
        assert type(c) == torch.Tensor
        
        self.dimension = c.shape[0]
        
        if len(c.shape) != 1:
            raise ValueError(f'The center should be a column tensor, but len(c.shape) is {len(c.shape)}')
        if G.numel() != 0 and self.dimension != G.shape[0]:
            raise ValueError(f'Dimension mismatch between center ({self.dimension}) and dependent generator matrix ({G.shape[0]}).')
        if Grest.numel() != 0 and self.dimension != Grest.shape[0]:
            raise ValueError(f'Dimension mismatch between center ({self.dimension}) and dependent generator matrix ({Grest.shape[0]}).')


        self.c = c
        self.G = G.reshape(self.dimension,-1)
        self.Grest = Grest.reshape(self.dimension,-1)

        if expMat == None and id == None:
            self.expMat = torch.eye(G.shape[1])
            self.id = torch.arange(G.shape[1])
        else:
            #check correctness of user input
            if not torch.all(expMat.floor() == expMat) or not torch.all(expMat >= 0) or expMat.shape[1] != G.shape[1]:
                raise ValueError('Invalid exponenet matrix.')
            expMat,G = removeRedundantExponents(expMat,G)
            self.G =G
            self.expMat =expMat
            if id != None:
                if id.shape[0] != expMat.shape[0]:
                    raise ValueError(f'Invalid vector of identifiers. Dimension of exponents matrix is {expMat.shape}')
                self.id = id
            else:
                self.id = torch.arange(expMat.shape[0])

    def  __add__(self,other):  
        '''
        Overloaded '+' operator for Minkowski sum
        '''
        # TODO: allow to add bw polyZonotope and Zonotope

        # if other is a vector
        if type(other) == np.ndarray or type(other) == torch.tensor:
            if type(other) == np.ndarray:
                other = torch.from_numpy(other)
            assert other.shape == self.c.shape:
            self.c + other
        
        # if other is a zonotope


        # if other is a polynomial zonotope
        elif type(other) == polyZonotope:
            assert other.dimension == self.dimension
            self.c += other.c
            self.G = torch.hstack((self.G,other.G))
            self.expMat = torch.block_diag(self.expMat,other.expMat)
            self.Grest = torch.hstack((self.Grest,other.Grest))
            self.id = torch.hstack((self.id,other.id+max(self.id)+1))

    def __matmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a matrix or an interval matrix with a polyZonotope

        '''
        # TODO: Need to define intervals

        # if other is a matrix
        if type(other) == np.ndarray or type(other) == torch.tensor:
            if type(other) == np.ndarray:
                other = torch.from_numpy(other)
            self.c = other@self.c

    








        