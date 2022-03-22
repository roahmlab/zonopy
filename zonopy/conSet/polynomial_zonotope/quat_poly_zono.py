"""
Define class for quaternion polynomial zonotopequaternion 
Reference: CORA
Writer: Yongseok Kwon
"""

from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope
from zonopy.conSet.polynomial_zonotope.utils import removeRedundantExponents, mergeExpMatrix


import torch
import numpy as np

EMPTY_TENSOR = torch.tensor([])
class quatPolyZonotope:
    '''
    qpZ: <quatPolyZonotope>
    
    c: <torch.Tensor> center of the polyonmial zonotope
    , shape: [4,1] 
    G: <torch.Tensor> generator matrix containing the dependent generators
    , shape: [4, N]
    Grest: <torch.Tensor> generator matrix containing the independent generators
    , shape: [4, M]
    expMat: <troch.Tensor> matrix containing the exponents for the dependent generators
    , shape: [p, N]
    id: <torch.Tensor> vector containing the integer identifiers for the dependent factors
    , shape: [p]

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
        assert c.shape[0] == 4
        self.dimension = 4
        
        if len(c.shape) != 1:
            raise ValueError(f'The center should be a column tensor, but len(c.shape) is {len(c.shape)}')
        if G.numel() != 0 and self.dimension != G.shape[0]:
            raise ValueError(f'Dimension mismatch between center ({self.dimension}) and dependent generator matrix ({G.shape[0]}).')
        if Grest.numel() != 0 and self.dimension != Grest.shape[0]:
            raise ValueError(f'Dimension mismatch between center ({self.dimension}) and dependent generator matrix ({Grest.shape[0]}).')
        if expMat is not None and expMat.numel() == 0:
            expMat = None
        
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
        elif id.numel() == 0:
            self.expMat = torch.eye(0,dtype=int)
            self.id = id
        else:

            raise ValueError('Identifiers can only be defined as long as the exponent matrix is defined.')
    def __str__(self):
        return f'\ncenter: \n {self.c} \n\n dependent generators: \n {self.G} \n\n exponent matrix: \n {self.expMat[self.id]} \n\n independent generators: \n {self.Grest} \n\n dimension: {self.dimension} \n'

    def  __add__(self,other):  
        '''
        Overloaded '+' operator for Minkowski sum
        self: <polyZonotope>
        other: <np.ndarray> or <torch.tensor> OR <zonotope> OR <polyZonotope>
        return <polyZonotope>
        '''
        # TODO: allow to add bw polyZonotope and zonotope

        # if other is a vector
        if type(other) == np.ndarray or type(other) == torch.Tensor:
            if type(other) == np.ndarray:
                other = torch.from_numpy(other)
            assert other.shape == self.c.shape
            c = self.c + other
            G, Grest, expMat, id = self.G, self.Grest, self.expMat, self.id
        
        # if other is a zonotope


        # if other is a polynomial zonotope
        elif type(other) == quatPolyZonotope:
            assert other.dimension == self.dimension
            c = self.c + other.c
            G = torch.hstack((self.G,other.G))
            Grest = torch.hstack((self.Grest,other.Grest))
            expMat = torch.block_diag(self.expMat,other.expMat)
            id = torch.hstack((self.id,other.id+self.id.numel()))

        return quatPolyZonotope(c,G,Grest,expMat,id)
    
    __radd__ = __add__

    def __invert__(self):
        c = torch.clone(self.c)
        G = torch.clone(self.G)
        Grest = torch.clone(self.Grest)
        c[1:], G[1:], Grest[1:] = -c[1:], -G[1:], -Grest[1:]
        expMat = torch.clone(self.expMat)
        id = torch.clone(self.id)
        return quatPolyZonotope(c,G,Grest,expMat,id)
    
    def __matmul__(self,other):
        '''
        Overloaded '@' operator for the rotation of a __ about a quatPolyZonotope
        self: <quatPolyZonotope>
        other: <torch.tensor> OR <polyZonotope>
        return <polyZonotope>
        
        other: <quatPolyZonotope>
        return <quatPolyZonotope>
        '''
        if type(other) == torch.Tensor:
            assert len(other.shape) == 1, 'The other object should be 1-D tensor.'  
            assert other.shape[0] == 3, 'Dimension of the other object should be 3.'            
            c = quat_rot(self.c, other, True)
            G = quat_rot(self.G,other,keep_axis_comp=False)
            Grest = quat_rot(self.Grest,other,keep_axis_comp=False)
            id = self.id
            expMat = self.expMat 
            return polyZonotope(c,G,Grest,expMat,id)
        
        # NOTE: this is 'OVERAPPROXIMATED' multiplication for keeping 'fully-k-sliceables'
        # The actual multiplication should take
        # dep. gnes.: C_G, G_c, G_G, Grest_Grest, G_Grest, Grest_G
        # indep. gens.: C_Grest, Grest_c
        #
        # But, the sliceable multiplication takes
        # dep. gnes.: C_G, G_c, G_G (fully-k-sliceable)
        # indep. gnes.: C_Grest, Grest_c, Grest_Grest
        #               G_Grest, Grest_G (partially-k-sliceable)
        
        elif type(other) == polyZonotope:
            assert other.dimension == 3, 'Dimension of the other object should be 3.' 
            id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            # NOTE: expMat shape?
            G = Grest = expMat = EMPTY_TENSOR
            
            c = quat_rot(self.c, other.c, True)
            
            # deal with dependent generators
            if other.G.numel() != 0:
                c_G = quat_rot(self.c, other.G)
                G = torch.hstack((G,c_G))
                expMat = torch.hstack((expMat,expMat2))
            if self.G.numel() != 0:
                G_c = quat_rot(self.G, other.c,keep_axis_comp=False)
                G = torch.hstack((G,G_c))
                expMat = torch.hstack((expMat,expMat1))
            if self.G.numel() != 0 and other.G.numel() != 0:
                G_G = quat_rot(self.G, other.G,keep_axis_comp=False)
                G = torch.hstack((G,G_G))
                # NOTE:
                for i in range(self.G.shape[-1]):
                    expMat = torch.hstack((expMat, expMat1[:,i].reshape(-1,1)+expMat2))
            
            # deal with independent generators
            if other.Grest.numel() != 0:
                c_Grest = quat_rot(self.c, other.Grest)
                Grest = torch.hstack((Grest,c_Grest))
            if self.Grest.numel() != 0:
                Grest_c = quat_rot(self.Grest,other.c,keep_axis_comp=False)
                Grest = torch.hstack((Grest,Grest_c))
            if self.Grest.numel() != 0 and other.Grest.numel() != 0:
                Grest_Grest = quat_rot(self.Grest,other.Grest,keep_axis_comp=False)
                Grest = torch.hstack((Grest,Grest_Grest))
            if self.G.numel() !=0 and other.Grest.numel() !=0:
                G_Grest = quat_rot(self.G,other.Grest,keep_axis_comp=False)
                Grest = torch.hstack((Grest,G_Grest))
            if self.Grest.numel() != 0 and other.G.numel() !=0:
                Grest_G = quat_rot(self.Grest,other.G,keep_axis_comp=False)
                Grest = torch.hstack((Grest,Grest_G))

            return polyZonotope(c,G,Grest,expMat,id)

    def __mul__(self,other):
        '''
        Overloaded '*' operator for the multiplication of a __ with a quatPolyZonotope
        self: <quatPolyZonotope>
        other: <torch.tensor> OR <polyZonotope>
        return <polyZonotope>
        
        other: <quatPolyZonotope>
        return <quatPolyZonotope>
        '''
        assert type(other) == quatPolyZonotope, 'The other object should be quaternion polynomial zonotope.'
        id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
        G = Grest = expMat = EMPTY_TENSOR
        c = quat_mul(self.c,other.c, True)
        
        # deal with dependent generators
        if other.G.numel() != 0:
            c_G = quat_mul(self.c, other.G)
            G = torch.hstack((G,c_G))
            expMat = torch.hstack((expMat,expMat2))
        if self.G.numel() != 0:
            G_c = quat_mul(self.G,other.c)
            G = torch.hstack((G,G_c))
            expMat = torch.hstack((expMat,expMat1))
        if self.G.numel() != 0 and other.G.numel() != 0:
            G_G = quat_mul(self.G,other.G)
            G = torch.hstack((G,G_G))
            # NOTE:
            for i in range(self.G.shape[-1]):
                expMat = torch.hstack((expMat, expMat1[:,i].reshape(-1,1)+expMat2))
        
        # deal with independent generators
        if other.Grest.numel() != 0:
            c_Grest = quat_mul(self.c, other.Grest)
            Grest = torch.hstack((Grest,c_Grest))
        if self.Grest.numel() != 0:
            Grest_c = quat_mul(self.Grest,other.c)
            Grest = torch.hstack((Grest,Grest_c))
        if self.Grest.numel() != 0 and other.Grest.numel() != 0:
            Grest_Grest = quat_mul(self.Grest,other.Grest)
            Grest = torch.hstack((Grest,Grest_Grest))
        if self.G.numel() !=0 and other.Grest.numel() !=0:
            G_Grest = quat_mul(self.G,other.Grest)
            Grest = torch.hstack((Grest,G_Grest))
        if self.Grest.numel() != 0 and other.G.numel() !=0:
            Grest_G = quat_mul(self.Grest,other.G)
            Grest = torch.hstack((Grest,Grest_G))

        return quatPolyZonotope(c,G,Grest,expMat,id)

def quat_mul(q1,q2,flat=False):
    '''
    q1, q2 shape: [4] or [4,n>1]
    r1, r2 shape: [n1*n2]
    v1, v2 shape: [3,n1*n2]
    '''
    assert q1.shape[0] == 4 and q2.shape[0] == 4
    r1, v1 = q1[0].reshape(-1), q1[1:].reshape(3,-1)
    r2, v2 = q2[0].reshape(-1), q2[1:].reshape(3,-1)
    
    n1, n2 = len(r1), len(r2)
    r1, v1 = r1.repeat_interleave(n2), v1.repeat_interleave(n2,dim=-1)
    r2, v2 = r2.repeat(n1), v2.repeat(1,n1)

    R = r1*r2 - torch.sum(v1*v2,dim=0) # [n1*n2]
    V = r1*v2 + r2*v1 + torch.cross(v1,v2) # [3,n1*n2]
    Q = torch.vstack((R,V))
    if Q.shape[1] == 1 and flat:
        Q = Q.reshape(4)
    return Q

def quat_rot(q1,q2,flat=False,keep_axis_comp=True):
    '''
    q1, q2 shape: [4] or [4,n>1]
    r1, r2 shape: [n1*n2]
    v1, v2 shape: [3,n1*n2]
    '''
    assert q1.shape[0] == 4 and q2.shape[0] == 3
    r1, v1 = q1[0].reshape(-1), q1[1:].reshape(3,-1)
    v2 = q2.reshape(3,-1)

    n1, n2 = len(r1), v2.shape[1]
    r1, v1 = r1.repeat_interleave(n2), v1.repeat_interleave(n2,dim=-1)
    v2 = v2.repeat(1,n1)

    if keep_axis_comp:
        r3 = - torch.sum(v1*v2,dim=0) # [n1*n2]
    else:
        r3 = torch.zeros(n1*n2)
    v3 = r1*v2 + torch.cross(v1,v2) # [3,n1*n2]
    # R = r3*r1 + torch.sum(v3*v1,dim=0)
    V = - r3*v1 + r1*v3 - torch.cross(v3,v1)
    if V.shape[1] == 1 and flat:
        V = V.reshape(3)
    return V