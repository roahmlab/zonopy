"""
Define class for polynomial zonotope
Reference: CORA
Writer: Yongseok Kwon
"""
from utils import removeRedundantExponents, mergeExpMatrix
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
    , shape: [p]
    # NOTE: may want string list of id

    # NOTE: need to be confirmed
    Eq. (coeff. a1,a2,...,aN; b1,b2,...,bp \in [0,1])
    G = [gd1,gd2,...,gdN]
    Grest = [gi1,gi2,...,giM]
    expMat = [[i11,i12,...,i1N],[i21,i22,...,i2N],...,[ip1,ip2,...,ipN]]
    id = [0,1,2,...,p-1]

    pZ = c + a1*gi1 + a2*gi2 + ... + aN*giN + b1^i11*b2^i21*...*bp^ip1*gd1 + b1^i12*b2^i22*...*bp^ip2*gd2 + ... 
    + b1^i1M*b2^i2M*...*bp^ipM*gdM
    
    '''
    def __init__(self,c=EMPTY_TENSOR,G=EMPTY_TENSOR,Grest=EMPTY_TENSOR,expMat=None,id=None,device='cpu'):
        
        # TODO: need to allow np.ndarray type
        # TODO: assign device
        # TODO: assign dtype for ind, exp
        # TODO: ind might be better to be list


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
        self.G = G.reshape(self.dimension,-1 if G.numel() != 0 else 0)
        self.Grest = Grest.reshape(self.dimension,-1 if Grest.numel() != 0 else 0)

        if expMat == None and id == None:
            self.expMat = torch.eye(self.G.shape[-1],dtype=int) # if G is EMPTY_TENSOR, it will be EMPTY_TENSOR, size = (0,0)
            self.id = torch.arange(self.G.shape[-1]) # if G is EMPTY_TENSOR, if will be EMPTY_TENSOR
        elif expMat != None:
            #check correctness of user input
            if not torch.all(expMat.to(dtype=int)==expMat) or not torch.all(expMat >= 0) or expMat.shape[1] != G.shape[-1]:
                raise ValueError('Invalid exponenet matrix.')
            expMat,G = removeRedundantExponents(expMat,G)
            self.G =G
            self.expMat =expMat
            if id != None:
                if id.shape[0] != expMat.shape[0]:
                    raise ValueError(f'Invalid vector of identifiers. Dimension of exponents matrix is {expMat.shape}')
                self.id = id.to(dtype=int)  
            else:
                self.id = torch.arange(expMat.shape[0])
        else:
            raise ValueError('Identifiers can only be defined as long as the exponent matrix is defined.')

    def  __add__(self,other):  
        '''
        Overloaded '+' operator for Minkowski sum
        self: <polyZonotope>
        other: <np.ndarray> or <torch.tensor> OR <zonotope> OR <polyZonotope>
        return <polyZonotope>
        '''
        # TODO: allow to add bw polyZonotope and zonotope

        # if other is a vector
        if type(other) == np.ndarray or type(other) == torch.tensor:
            if type(other) == np.ndarray:
                other = torch.from_numpy(other)
            assert other.shape == self.c.shape
            c = self.c + other
            G, Grest, expMat, id = self.G, self.Grest, self.expMat, self.id
        
        # if other is a zonotope


        # if other is a polynomial zonotope
        elif type(other) == polyZonotope:
            assert other.dimension == self.dimension
            c = self.c + other.c
            G = torch.hstack((self.G,other.G))
            Grest = torch.hstack((self.Grest,other.Grest))
            expMat = torch.block_diag(self.expMat,other.expMat)
            id = torch.hstack((self.id,other.id+self.id.numel()))

        return polyZonotope(c,G,Grest,expMat,id)
    
    __radd__ = __add__

    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a matrix or an interval matrix with a polyZonotope
        self: <polyZonotope>
        other: <np.ndarray> or <torch.tensor> OR <intervals>
        return <polyZonotope>
        '''
        # TODO: Need to define intervals for matrix
        
        # if other is a matrix
        if type(other) == np.ndarray or type(other) == torch.tensor:
            if type(other) == np.ndarray:
                other = torch.from_numpy(other)
            
            c = other@self.c
            if self.G.numel() != 0:
                G = other@self.G

            if self.Grest.numel() != 0:
                Grest = other@self.Grest

        # if other is an interval matrix

        return polyZonotope(c,G,Grest,self.expMat,self.id)
    
    #def reduce(self,option,order,*args,**kwargs):
        '''
        '''

        #if option == 'adaptive':
            #pZ = reduceAdaptive(self,order)
            #return pZ
            
             

        
    def exactPlus(self,other):
        '''
        compute the addition of two sets while preserving the dependencies between the two sets
        self: <polyZonotope>
        other: <polyZonotope>
        return <polyZonotope>
        '''
        # NOTE: need to write mergeExpMatrix
        id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
        
        ExpNew, Gnew = removeRedundantExponents(
            torch.hstack((expMat1, expMat2)),
            torch.hstack((self.G, other.G))
            )
        c = self.c + other.c
        Grest = torch.hstack((self.Grest,other.Grest))
        return polyZonotope(c,Gnew,Grest,ExpNew,id)

    def cartProd(self,other=None):
        '''
        compute teh cartesian product of two polyZonotopes
        self: <polyZonotope>
        other: <polyZonotope>
        return <polyZonotope>
        '''    
        if other == None:
            return self
        
        # convert other set representations to polyZonotopes (other)
        if type(other) != polyZonotope:
            if type(other) == zonotope or type(other) == interval:
                
                pZ2 = zonotope(pZ2)
                pZ2 = polyZonotope(pZ2.c)

            
            elif type(other) == np.ndarray or type(other) == torch.Tensor:
                other = polyZonotope()

    def cross(self,other):
        '''
        
        '''
        if type(other) == torch.Tensor:
            A = torch.tensor([[0,-other[2],other[1]],[other[2],0,-other[1]],[-other[1],other[0],0]])
        else:
            c = other.c
            g = other.G
            Z = torch.hstack((c,g))
            G = torch.zeros(Z.shape[1]-1,3,3)
            for j in range(Z.shape[1]):
                z = Z[:,j]
                M = torch.tensor([[0,-z[2],z[1]],[z[2],0,-z[1]],[-z[1],z[0],0]])
                if j == 0:
                    C = M
                else:
                    G[j-1] = M
            A = matPolyZonotope(C=C,G=G,expMat=other.expMat,id=other.id)
        
        return A@self

if __name__ == '__main__':
    c1 = torch.tensor([1,2])
    Grest1 = torch.tensor([[1,2,3],[1,4,6]])
    G1 = torch.tensor([[7,8,3,1,2],[1,8,2,1,5]])
    expMat1 = torch.tensor([[1,0,3,2,1],[0,0,0,2,0],[1,2,4,3,1]])
    c2 = torch.tensor([2,2])
    pz1 = polyZonotope(c1,G1,Grest1,expMat1)
    pz2 = polyZonotope(c2)
    pz =pz1 + pz2
    print(pz.id)

    pz = pz1.exactPlus(pz2)
    print(pz1.c)
    print(pz.c)
    print(pz.G)
    print(pz.expMat)
    print(pz.id)








        