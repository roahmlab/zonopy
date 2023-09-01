"""
Define class for matrix polynomial zonotope
Author: Yongseok Kwon
Reference: Patrick Holme's implementation
"""
from zonopy.conSet.polynomial_zonotope.utils import removeRedundantExponents
from zonopy import polyZonotope
import zonopy as zp
import torch
import numpy as np
import zonopy.internal as zpi
from ..gen_ops import (
    _matmul_genmpz_impl,
    )

class matPolyZonotope():
    '''
    <matPolyZonotope>

    c: <torch.Tensor> center maxtrix of the matrix polyonmial zonotope
    , shape: [nx,ny] 
    G: <torch.tensor> generator tensor containing the dependent generators 
    , shape: [N, nx, ny] 
    Grest: <torch.Tensor> generator tensor containing the independent generators
    , shape: [M, nx, ny]
    expMat: <troch.Tensor> matrix containing the exponents for the dependent generators
    , shape: [N, p]
    id: <torch.Tensor> vector containing the integer identifiers for the dependent factors
    , shape: [p]
    compress: <int> level for compressing dependent generators with expodent
    0: no compress, 1: compress zero dependent generators, 2: compress zero dependent generators and remove redundant expodent

    Eq. (coeff. a1,a2,...,aN; b1,b2,...,bp \in [0,1])
    G = [[Gd1],[Gd2],...,[GdN]]
    Grest = [[Gi1],[Gi2],...,[GiM]]
    (Gd1,Gd2,...,GdN,Gi1,Gi2,...,GiM \in R^(nx,ny), matrix)
    expMat = [[i11,i12,...,i1p],[i21,i22,...,i2p],...,[iN1,iN2,...,iNp]]
    id = [0,1,2,...,p-1]

    pZ = c + a1*Gi1 + a2*Gi2 + ... + aN*GiN + b1^i11*b2^i21*...*bp^ip1*Gd1 + b1^i12*b2^i22*...*bp^ip2*Gd2 + ... 
    + b1^i1M*b2^i2M*...*bp^ipM*GdM
    '''
    def __init__(self,Z,n_dep_gens=0,expMat=None,id=None,copy_Z=True, dtype=None, device=None):
        # If compress=2, it will always copy.

        # Make sure Z is a tensor
        if not isinstance(Z, torch.Tensor) and dtype is None:
            dtype = torch.float
        Z = torch.as_tensor(Z, dtype=dtype, device=device)

        # Make an expMat and id if not given
        if expMat is None and id is None:
            self.expMat = torch.eye(n_dep_gens,dtype=torch.long,device=Z.device)
            self.id = np.arange(self.expMat.shape[1],dtype=int)

        # Otherwise make sure expMat is right
        elif expMat is not None:
            if not isinstance(expMat, torch.Tensor):
                expMat = torch.as_tensor(expMat,dtype=torch.long,device=Z.device)
            assert expMat.shape[0] == n_dep_gens, 'Invalid exponent matrix.' 
            if zpi.__debug_extra__: assert torch.all(expMat >= 0), 'Invalid exponent matrix.'
            
            self.expMat = expMat
                
            # Make sure ID is right
            if id is not None:
                self.id = np.asarray(id, dtype=int).flatten()
            else:
                self.id = np.arange(self.expMat.shape[1],dtype=int)
                
        # Otherwise ID is given, but not the expMat, so make identity
        else:
            self.id = np.asarray(id, dtype=int).flatten()
            assert len(self.id) == n_dep_gens, 'Number of dependent generators must match number of id\'s!'
            self.expMat = torch.eye(n_dep_gens,dtype=torch.long,device=Z.device)
        
        # Copy the Z if requested
        if copy_Z:
            self.Z = torch.clone(Z)
        # Or save it itself
        else:
            self.Z = Z
        self.n_dep_gens = n_dep_gens

    def compress(self, compression_level):
        # Remove zero generators
        if compression_level == 1:
            nonzero_g = torch.sum(self.G!=0,(-1,-2))!=0 # non-zero generator index
            G = self.G[nonzero_g]
            expMat = self.expMat[nonzero_g]

        # Remove generators related to redundant exponents
        elif compression_level == 2: 
            expMat, G = removeRedundantExponents(self.expMat, self.G)

        else:
            raise ValueError("Can only compress to 1 or 2!")

        # Update self
        self.Z = torch.vstack((self.Z[0].unsqueeze(0), G, self.Z[1+self.n_dep_gens:]))
        self.expMat = expMat
        self.n_dep_gens = G.shape[0]

        # For chaining
        return self

    @property
    def dtype(self):
        return self.Z.dtype
    @property
    def itype(self):
        return self.expMat.dtype
    @property
    def device(self):
        return self.Z.device
    @property 
    def C(self):
        return self.Z[0]
    @property 
    def G(self):
        return self.Z[1:self.n_dep_gens+1]
    @property 
    def Grest(self):
        return self.Z[self.n_dep_gens+1:]
    @property
    def n_generators(self):
        return len(self.Z)-1
    @property
    def n_indep_gens(self):
        return len(self.Z)-self.n_dep_gens-1
    @property 
    def n_rows(self):
        return self.Z.shape[1]
    @property 
    def n_cols(self):
        return self.Z.shape[2]
    @property 
    def shape(self):
        return self.Z.shape[-2:]
    @property
    def T(self):        
        return matPolyZonotope(self.Z.transpose(1,2),self.n_dep_gens,self.expMat,self.id,copy_Z=False)
    @property 
    def input_pairs(self):
        # id_sorted, order = torch.sort(self.id)
        order = np.argsort(self.id)
        expMat_sorted = self.expMat[:,order] 
        # return self.Z, self.n_dep_gens, expMat_sorted, id_sorted
        return self.Z, self.n_dep_gens, expMat_sorted, self.id[order]
        
    def to(self,dtype=None,itype=None,device=None):
        Z = self.Z.to(dtype=dtype,device=device, non_blocking=True)
        expMat = self.expMat.to(dtype=itype,device=device, non_blocking=True)
        # id = self.id.to(device=device)
        return matPolyZonotope(Z,self.n_dep_gens,expMat,self.id,copy_Z=False)
    def cpu(self):
        Z = self.Z.cpu()
        expMat = self.expMat.cpu()
        # id = self.id.cpu()
        return matPolyZonotope(Z,self.n_dep_gens,expMat,self.id,copy_Z=False)
        
    def __matmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a __ with a matPolyZonotope
        self: <matPolyZonotope>
        other: <torch.tensor> OR <polyZonotope>
        return <polyZonotope>
        
        other: <matPolyZonotope>
        return <matPolyZonotope>
        '''
        if isinstance(other, torch.Tensor):
            assert other.shape[0] == self.n_cols or other.shape[-2] == self.n_cols
            
            if len(other.shape) == 1:
                Z = self.Z @ other
                return polyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False).compress(1)
            elif len(other.shape) == 2:
                Z = self.Z @ other
                return matPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False).compress(1)
            else:
                Z = self.Z @ other.unsqueeze(-3)
                return zp.batchMatPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False).compress(1)

        # NOTE: this is 'OVERAPPROXIMATED' multiplication for keeping 'fully-k-sliceables'
        # The actual multiplication should take
        # dep. gnes.: C_G, G_c, G_G, Grest_Grest, G_Grest, Grest_G
        # indep. gens.: C_Grest, Grest_c
        #
        # But, the sliceable multiplication takes
        # dep. gnes.: C_G, G_c, G_G (fully-k-sliceable)
        # indep. gnes.: C_Grest, Grest_c, Grest_Grest
        #               G_Grest, Grest_G (partially-k-sliceable)
        
        elif isinstance(other,polyZonotope):
            # Shim other to a matPolyZonotope
            shim_other = matPolyZonotope(other.Z.unsqueeze(-1), other.n_dep_gens, other.expMat, other.id, copy_Z=False)
            Z, n_dep_gens, expMat, id = _matmul_genmpz_impl(self, shim_other)
            return polyZonotope(Z.squeeze(-1), n_dep_gens, expMat, id).compress(2)

        elif isinstance(other,matPolyZonotope):
            args = _matmul_genmpz_impl(self, other)
            return matPolyZonotope(*args).compress(2)
        
        else:
            return NotImplemented

    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a __ with a matPolyZonotope
        self: <matPolyZonotope>
        other: <torch.tensor> OR <polyZonotope>
        return <polyZonotope>
        
        other: <matPolyZonotope>
        return <matPolyZonotope>
        '''
        if isinstance(other,torch.Tensor):
            assert len(other.shape) == 2, 'The other object should be 2-D tensor.'  
            assert other.shape[1] == self.n_rows
            Z = other @ self.Z
            return matPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False).compress(1)
        
        elif isinstance(other,polyZonotope):
            # Shim other to a matPolyZonotope
            shim_other = zp.matPolyZonotope(other.Z.unsqueeze(-2),other.n_dep_gens,other.expMat,other.id,copy_Z=False)
            Z, n_dep_gens, expMat, id = _matmul_genmpz_impl(shim_other, self)
            return zp.polyZonotope(Z.squeeze(-2), n_dep_gens, expMat, id).compress(2)
        
        else:
            return NotImplemented

    def to_matZonotope(self):
        if self.n_dep_gens != 0:
            ind = torch.any(self.expMat%2,1)
            Gquad = self.G[~ind]
            c = self.C + 0.5*torch.sum(Gquad,0)
            Z = torch.vstack((c.unsqueeze(0), self.G[ind],0.5*Gquad,self.Grest))
        else:
            Z = self.Z
        return zp.matZonotope(Z)


    def reduce(self,order,option='girard'):
        # extract dimensions
        N = self.n_rows * self.n_cols
        P = self.n_dep_gens 
        Q = self.n_indep_gens
            
        # number of gens kept (N gens will be added back after reudction)
        K = int(N*order-N)
        # check if the order need to be reduced
        if P+Q > N*order and K >=0:
            G = self.Z[1:]
            # half the generators length for exponents that are all even
            temp = torch.any(self.expMat%2,1)
            ind = (~temp).nonzero().squeeze()
            G[ind] *= 0.5
            # caculate the length of the gens with a special metric
            len = torch.sum(G**2,(1,2))
            # determine the smallest gens to remove            
            ind = torch.argsort(len,descending=True)
            ind_red,ind_rem = ind[:K], ind[K:]
            # split the indices into the ones for dependent and independent
            indDep = ind_rem[ind_rem < P]
            ind_REM = torch.hstack((indDep, ind_rem[ind_rem >= P]))
            indDep_red = ind_red[ind_red < P]
            ind_RED = torch.hstack((indDep_red,ind_red[ind_red >= P]))
            # construct a zonotope from the gens that are removed
            n_dg_rem = indDep.shape[0]
            Erem = self.expMat[indDep]
            Ztemp = torch.vstack((torch.zeros(1,self.n_rows,self.n_cols,dtype=self.dtype,device=self.device),G[ind_REM]))
            pZtemp = matPolyZonotope(Ztemp,n_dg_rem,Erem,self.id).compress(1) # NOTE: ID???
            zono = pZtemp.to_matZonotope() # zonotope over-approximation
            # reduce the constructed zonotope with the reducetion techniques for linear zonotopes
            zonoRed = zono.reduce(1,option)
            
            # remove the gens that got reduce from the gen matrices
            expMatRed = self.expMat[indDep_red]    
            n_dg_red = indDep_red.shape[0]
            # add the reduced gens as new indep gens
            ZRed = torch.vstack(((self.C + zonoRed.center).unsqueeze(0),G[ind_RED],zonoRed.generators))
        else:
            ZRed = self.Z
            n_dg_red = self.n_dep_gens
            expMatRed = self.expMat
        # remove all exponent vector dimensions that have no entries
        ind = (torch.sum(expMatRed,0)>0).cpu().numpy()
        #ind = temp.nonzero().reshape(-1)
        expMatRed = expMatRed[:,ind]
        idRed = self.id[ind]
        if self.n_rows == 1 and self.n_cols == 1:
            ZRed = torch.vstack((ZRed[:1],ZRed[1:n_dg_red+1].sum(0).unsqueeze(0),ZRed[n_dg_red+1:]))
        return matPolyZonotope(ZRed,n_dg_red,expMatRed,idRed,copy_Z=False).compress(1)

    def reduce_indep(self,order,option='girard'):
        # extract dimensions
        N = self.n_rows * self.n_cols
        P = self.n_dep_gens 
        Q = self.n_indep_gens
            
        # number of gens kept (N gens will be added back after reudction)
        K = int(N*order-N)
        # check if the order need to be reduced
        if Q > N*order and K >=0:
            G = self.Grest
            # caculate the length of the gens with a special metric
            len = torch.sum(G**2,(1,2))
            # determine the smallest gens to remove            
            ind = torch.argsort(len,descending=True)
            ind_red,ind_rem = ind[:K], ind[K:]
            # reduce the generators with the reducetion techniques for linear zonotopes            
            d = torch.sum(abs(G[ind_rem]),0).reshape(-1)
            Gbox = torch.diag(d).reshape(-1,self.n_rows,self.n_cols)
            # add the reduced gens as new indep gens
            ZRed = torch.vstack((self.C .unsqueeze(0),self.G,G[ind_red],Gbox))
        else:
            ZRed = self.Z
        n_dg_red = self.n_dep_gens
        if self.n_rows == 1 == self.n_cols and n_dg_red != 1:
            ZRed = torch.vstack((ZRed[:1],ZRed[1:n_dg_red+1].sum(0).unsqueeze(0),ZRed[n_dg_red+1:]))
            n_dg_red = 1
        return matPolyZonotope(ZRed,n_dg_red,self.expMat,self.id,copy_Z=False).compress(1)
    
    @staticmethod
    def zeros(dim1, dim2 = None):
        dim2 = dim1 if dim2 is not None else dim2
        Z = torch.zeros((1, dim1, dim2))
        expMat = torch.empty((0,0),dtype=torch.int64)
        id = np.empty(0,dtype=np.int64)
        return zp.matPolyZonotope(Z, 0, expMat=expMat, id=id, copy_Z=False)
    
    @staticmethod
    def ones(dim1, dim2 = None):
        dim2 = dim1 if dim2 is not None else dim2
        Z = torch.zeros((1, dim1, dim2))
        expMat = torch.empty((0,0),dtype=torch.int64)
        id = np.empty(0,dtype=np.int64)
        return zp.matPolyZonotope(Z, 0, expMat=expMat, id=id, copy_Z=False)
    
    @staticmethod
    def eye(dim):
        Z = torch.eye(dim).unsqueeze(0)
        expMat = torch.empty((0,0),dtype=torch.int64)
        id = np.empty(0,dtype=np.int64)
        return zp.matPolyZonotope(Z, 0, expMat=expMat, id=id, copy_Z=False)

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
    flag = zp.close(result1.to_zonotope(),result2.to_zonotope())
    print(flag)
