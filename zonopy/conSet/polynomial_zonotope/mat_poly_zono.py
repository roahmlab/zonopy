"""
Define class for matrix polynomial zonotope
Reference: Patrick Holme's matPolyZonotope
Writer: Yongseok Kwon
"""
from zonopy.conSet.polynomial_zonotope.utils import removeRedundantExponents, mergeExpMatrix
from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope
from zonopy.conSet.utils import G_mul_c, G_mul_g, C_mul_G, G_mul_C, G_mul_G
import torch
EMPTY_TENSOR = torch.tensor([])

class matPolyZonotope():
    '''
    <matPolyZonotope>

    c: <torch.Tensor> center maxtrix of the matrix polyonmial zonotope
    , shape: [nx,ny,1] 
    G: <torch.tensor> generator tensor containing the dependent generators 
    , shape: [nx, ny, N] 
    Grest: <torch.Tensor> generator tensor containing the independent generators
    , shape: [nx, ny, M]
    expMat: <troch.Tensor> matrix containing the exponents for the dependent generators
    , shape: [p, N]
    id: <torch.Tensor> vector containing the integer identifiers for the dependent factors
    , shape: [p]

    Eq. (coeff. a1,a2,...,aN; b1,b2,...,bp \in [0,1])
    G = [Gd1,Gd2,...,GdN]
    Grest = [Gi1,Gi2,...,GiM]
    (Gd1,Gd2,...,GdN,Gi1,Gi2,...,GiM \in R^(nx,ny), matrix)
    expMat = [[i11,i12,...,i1N],[i21,i22,...,i2N],...,[ip1,ip2,...,ipN]]
    id = [0,1,2,...,p-1]

    pZ = c + a1*Gi1 + a2*Gi2 + ... + aN*GiN + b1^i11*b2^i21*...*bp^ip1*Gd1 + b1^i12*b2^i22*...*bp^ip2*Gd2 + ... 
    + b1^i1M*b2^i2M*...*bp^ipM*GdM
    '''
    def __init__(self,C,G=EMPTY_TENSOR,Grest=EMPTY_TENSOR,expMat=None,id=None):
        assert type(C) == torch.Tensor

        if len(C.shape) != 2:
            raise ValueError('The center should be a matrix tensor, but C.shape is '+str(C.shape).replace('torch.Size','')+'.')
        self.n_rows = C.shape[0]
        self.n_cols = C.shape[1]

        if G.numel() != 0 and (self.n_rows != G.shape[0] or self.n_cols != G.shape[1]):
            raise ValueError(f'Matrix dimension mismatch between center ([{self.n_rows}, {self.n_cols}]) and dependent generator matrix ([{G.shape[0]}, {G.shape[1]}]).')
        if Grest.numel() != 0 and (self.n_rows != Grest.shape[0] or self.n_cols != Grest.shape[1]):
            raise ValueError(f'Matrix dimension mismatch between center ([{self.n_rows}, {self.n_cols}]) and independent generator matrix ([{Grest.shape[0]}, {Grest.shape[1]}]).')
        if expMat is not None and expMat.numel() == 0:
            expMat = None

        self.C = C
        self.G = G.reshape(self.n_rows,self.n_cols,-1 if G.numel() != 0 else 0)
        self.Grest = Grest.reshape(self.n_rows,self.n_cols,-1 if G.numel() != 0 else 0)

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

    def __matmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a __ with a matPolyZonotope
        self: <matPolyZonotope>
        other: <torch.tensor> OR <polyZonotope>
        return <polyZonotope>
        
        other: <matPolyZonotope>
        return <matPolyZonotope>
        '''
        if type(other) == torch.Tensor:
            assert len(other.shape) == 1, 'The other object should be 1-D tensor.'  
            assert other.shape[0] == self.n_cols
            
            c = self.C @ other
            G = G_mul_c(self.G,other)
            Grest = G_mul_c(self.Grest,other)
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
            assert self.n_cols == other.dimension
            #id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            id = torch.hstack((self.id,other.id+self.id.numel()))
            expMat1 = torch.vstack((self.expMat,torch.zeros(other.expMat.shape[0],self.expMat.shape[1])))
            expMat2 = torch.vstack((torch.zeros(self.expMat.shape[0],other.expMat.shape[1]),other.expMat))
            # NOTE: expMat shape?
            G = Grest = expMat = EMPTY_TENSOR
            
            c = self.C @ other.c
            
            # deal with dependent generators
            if other.G.numel() != 0:
                C_G = self.C @ other.G
                G = torch.hstack((G,C_G))
                expMat = torch.hstack((expMat,expMat2))
            if self.G.numel() != 0:
                G_c = G_mul_c(self.G,other.c)
                G = torch.hstack((G,G_c))
                expMat = torch.hstack((expMat,expMat1))
            if self.G.numel() != 0 and other.G.numel() != 0:
                G_G = G_mul_g(self.G,other.G,self.n_rows)
                G = torch.hstack((G,G_G))
                for i in range(self.G.shape[-1]):
                    expMat = torch.hstack((expMat, expMat1[:,i].reshape(-1,1)+expMat2))
            # deal with independent generators
            if other.Grest.numel() != 0:
                C_Grest = self.C @ other.Grest
                Grest = torch.hstack((Grest,C_Grest))
            if self.Grest.numel() != 0:
                Grest_c = G_mul_c(self.Grest,other.c)
                Grest = torch.hstack((Grest,Grest_c))
            if self.Grest.numel() != 0 and other.Grest.numel() != 0:
                Grest_Grest = G_mul_g(self.Grest,other.Grest)
                Grest = torch.hstack((Grest,Grest_Grest))
            if self.G.numel() !=0 and other.Grest.numel() !=0:
                G_Grest = G_mul_g(self.G,other.Grest)
                Grest = torch.hstack((Grest,G_Grest))
            if self.Grest.numel() != 0 and other.G.numel() !=0:
                Grest_G = G_mul_g(self.Grest,other.G)
                Grest = torch.hstack((Grest,Grest_G))

            return polyZonotope(c,G,Grest,expMat,id)

        elif type(other) == matPolyZonotope:
            assert self.n_cols == other.n_rows
            #id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            id = torch.hstack((self.id,other.id+self.id.numel()))
            expMat1 = torch.vstack((self.expMat,torch.zeros(other.expMat.shape[0],self.expMat.shape[1])))
            expMat2 = torch.vstack((torch.zeros(self.expMat.shape[0],other.expMat.shape[1]),other.expMat))
            
            dims = [self.n_rows, self.n_cols, other.n_cols]
            G = Grest = expMat = EMPTY_TENSOR
            C = self.C @ other.C
            
            # deal with dependent generators
            if other.G.numel() != 0:
                C_G = C_mul_G(self.C, other.G,dims)
                G = torch.cat((G,C_G),dim=-1)
                expMat = torch.hstack((expMat,expMat2))
            if self.G.numel() != 0:
                G_C = G_mul_C(self.G,other.C)
                G = torch.cat((G,G_C),dim=-1)
                expMat = torch.hstack((expMat,expMat1))
            if self.G.numel() != 0 and other.G.numel() != 0:
                G_G = G_mul_G(self.G,other.G,dims)
                G = torch.cat((G,G_G),dim=-1)
                # NOTE:
                for i in range(self.G.shape[-1]):
                    expMat = torch.hstack((expMat, expMat1[:,i].reshape(-1,1)+expMat2))
            
            # deal with independent generators
            if other.Grest.numel() != 0:
                C_Grest = C_mul_G(self.C, other.Grest,dims)
                Grest = torch.cat((Grest,C_Grest),dim=-1)
            if self.Grest.numel() != 0:
                Grest_C = G_mul_C(self.Grest,other.C)
                Grest = torch.cat((Grest,Grest_C),dim=-1)
            if self.Grest.numel() != 0 and other.Grest.numel() != 0:
                Grest_Grest = G_mul_G(self.Grest,other.Grest,dims)
                Grest = torch.cat((Grest,Grest_Grest),dim=-1)
            if self.G.numel() !=0 and other.Grest.numel() !=0:
                G_Grest = G_mul_G(self.G,other.Grest,dims)
                Grest = torch.cat((Grest,G_Grest),dim=-1)
            if self.Grest.numel() != 0 and other.G.numel() !=0:
                Grest_G = G_mul_G(self.Grest,other.G,dims)
                Grest = torch.cat((Grest,Grest_G),dim=-1)
            
            return matPolyZonotope(C,G,Grest,expMat,id)


        else:
            raise ValueError('the other object should be torch tensor or polynomial zonotope.')

    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a __ with a matPolyZonotope
        self: <matPolyZonotope>
        other: <torch.tensor> OR <polyZonotope>
        return <polyZonotope>
        
        other: <matPolyZonotope>
        return <matPolyZonotope>
        '''
        if type(other) == torch.Tensor:
            assert len(other.shape) == 2, 'The other object should be 2-D tensor.'  
            assert other.shape[1] == self.n_rows
                
            C = other @ self.C
            G = other @ self.G
            Grest = other @ self.Grest
            return matPolyZonotope(C,G,Grest,self.expMat,self.id)

    def to_matZonotope(self):
        from zonopy.conSet.zonotope.mat_zono import matZonotope
        Z = torch.cat((self.C.reshape(self.n_rows,self.n_cols,1),self.G),-1)
        Z = torch.cat((Z,self.Grest),-1)
        return matZonotope(Z)

if __name__ == '__main__':
    
    C1 = torch.rand(3,3,dtype=torch.float)
    C2 = torch.rand(3,3,dtype=torch.float)
    G1 = torch.rand(3,3,4,dtype=torch.float)
    G2 = torch.rand(3,3,2,dtype=torch.float)
    Grest1 = torch.rand(3,3,2,dtype=torch.float)
    Grest2 = torch.rand(3,3,3,dtype=torch.float)
    mp1 = matPolyZonotope(C1,G1,Grest1)
    mp2 = matPolyZonotope(C2,G2,Grest2)

    cen = torch.rand(3,dtype=torch.float)
    gen = torch.rand(3,4,dtype=torch.float)
    grest = torch.rand(3,2,dtype=torch.float)
    pz = polyZonotope(cen,gen,grest)
    
    result1 = mp1@mp2@pz
    result2 = mp1@(mp2@pz)
    #import pdb; pdb.set_trace()
    import zonopy as zp
    flag = zp.close(result1.to_zonotope(),result2.to_zonotope())
    print(flag)
