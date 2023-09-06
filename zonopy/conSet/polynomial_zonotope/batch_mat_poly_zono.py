"""
Define class for matrix polynomial zonotope
Author: Yongseok Kwon
Reference: Patrick Holme's implementation
"""
from zonopy.conSet.polynomial_zonotope.utils import removeRedundantExponentsBatch
from zonopy import batchPolyZonotope
from zonopy import matPolyZonotope
import zonopy as zp
import torch
import numpy as np
import zonopy.internal as zpi
from ..gen_ops import (
    _matmul_genmpz_impl,
    )

class batchMatPolyZonotope():
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

    Eq. (coeff. a1,a2,...,aN; b1,b2,...,bp \in [0,1])
    G = [[Gd1],[Gd2],...,[GdN]]
    Grest = [[Gi1],[Gi2],...,[GiM]]
    (Gd1,Gd2,...,GdN,Gi1,Gi2,...,GiM \in R^(nx,ny), matrix)
    expMat = [[i11,i12,...,i1p],[i21,i22,...,i2p],...,[iN1,iN2,...,iNp]]
    id = [0,1,2,...,p-1]

    pZ = c + a1*Gi1 + a2*Gi2 + ... + aN*GiN + b1^i11*b2^i21*...*bp^ip1*Gd1 + b1^i12*b2^i22*...*bp^ip2*Gd2 + ... 
    + b1^i1M*b2^i2M*...*bp^ipM*GdM
    '''
    def __init__(self,Z,n_dep_gens=0,expMat=None,id=None,copy_Z=True, device=None, dtype=None):
        # If compress=2, it will always copy.

        # Make sure Z is a tensor and shaped right
        if not isinstance(Z, torch.Tensor) and dtype is None:
            dtype = torch.get_default_dtype()
        Z = torch.as_tensor(Z, dtype=dtype, device=device)
        assert len(Z.shape) > 3, f'The dimension of Z input should be either 1 or 2, not {len(Z.shape)}.'

        self.batch_dim = len(Z.shape) - 3
        self.batch_idx_all = tuple([slice(None) for _ in range(self.batch_dim)])

        # Make an expMat and id if not given
        if expMat is None and id is None:
            self.expMat = torch.eye(n_dep_gens,dtype=torch.long,device=Z.device) # if G is EMPTY_TENSOR, it will be EMPTY_TENSOR, size = (0,0)
            self.id = np.arange(self.expMat.shape[1],dtype=int)

        # Otherwise make sure expMat is right
        elif expMat is not None:
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
            nonzero_g = torch.sum(self.G!=0,tuple(range(self.batch_dim))+(-1,-2))!=0 # non-zero generator index
            G = self.G[...,nonzero_g,:,:]
            expMat = self.expMat[nonzero_g]

        # Remove generators related to redundant exponents
        elif compression_level == 2: 
            expMat, G = removeRedundantExponentsBatch(self.expMat, self.G, [], 3)

        else:
            raise ValueError("Can only compress to 1 or 2!")

        # Update self
        self.Z = torch.cat((self.C.unsqueeze(-3), G, self.Grest), dim=-3)
        self.expMat = expMat
        self.n_dep_gens = G.shape[-3]

        # For chaining
        return self
    
    def __getitem__(self,idx):
        Z = self.Z[idx]
        if len(Z.shape) > 3:
            return batchMatPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False)
        else:
            return zp.matPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False)
        
    # def __len__(self):
    #     return self.Z.shape[0]
    
    @property 
    def batch_shape(self):
        return self.Z.shape[:-3]
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
        return self.Z[self.batch_idx_all+(0,)]
    @property 
    def G(self):
        return self.Z[self.batch_idx_all+(slice(1,self.n_dep_gens+1),)]
    @property 
    def Grest(self):
        return self.Z[self.batch_idx_all+(slice(self.n_dep_gens+1,None),)]
    @property
    def n_generators(self):
        return self.Z.shape[-3]-1
    @property
    def n_indep_gens(self):
        return self.Z.shape[-3]-1-self.n_dep_gens
    @property
    def n_rows(self):
        return self.Z.shape[-2]
    @property
    def n_cols(self):
        return self.Z.shape[-1]
    @property 
    def shape(self):
        return self.Z.shape[-2:]
    @property
    def T(self):        
        return batchMatPolyZonotope(self.Z.transpose(-1,-2),self.n_dep_gens,self.expMat,self.id,copy_Z=False)
    @property 
    def input_pairs(self):
        id_sorted, order = torch.sort(self.id)
        expMat_sorted = self.expMat[:,order] 
        return self.Z, self.n_dep_gens, expMat_sorted, id_sorted
        
    def to(self,dtype=None,itype=None,device=None):
        Z = self.Z.to(dtype=dtype,device=device, non_blocking=True)
        expMat = self.expMat.to(dtype=itype,device=device, non_blocking=True)
        # id = self.id.to(device=device)
        return batchMatPolyZonotope(Z,self.n_dep_gens,expMat,self.id,copy_Z=False)
        
    def cpu(self):
        Z = self.Z.cpu()
        expMat = self.expMat.cpu()
        # id = self.id.cpu()
        return batchMatPolyZonotope(Z,self.n_dep_gens,expMat,self.id,copy_Z=False)

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
            
            if len(other.shape) == 1:
                Z = self.Z @ other
                return batchPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False).compress(1)
            elif len(other.shape) == 2:
                Z = self.Z @ other
                return batchMatPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False).compress(1)
            else:
                Z = self.Z @ other.unsqueeze(-3)
                return batchMatPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False).compress(1)

        elif isinstance(other, (batchPolyZonotope, zp.polyZonotope)):
            # Shim other to a batchMatPolyZonotope (and add an arbitrary batch dim to remove)
            shim_other = zp.batchMatPolyZonotope(other.Z.unsqueeze(-1).unsqueeze(0),other.n_dep_gens,other.expMat,other.id,copy_Z=False)
            Z, n_dep_gens, expMat, id = _matmul_genmpz_impl(self, shim_other)
            return batchPolyZonotope(Z.squeeze(-1).squeeze(0), n_dep_gens, expMat, id).compress(2)

        elif isinstance(other, (batchMatPolyZonotope, matPolyZonotope)):
            args = _matmul_genmpz_impl(self, other)
            return batchMatPolyZonotope(*args).compress(2)
        
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
            assert other.shape[-1] == self.n_rows
            assert len(other.shape) >= 2
            Z = other @ self.Z
            return batchMatPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id).compress(1)
        
        elif isinstance(other,(zp.polyZonotope,zp.batchMatPolyZonotope)):
            # Shim other to a batchMatPolyZonotope (and add an arbitrary batch dim to remove)
            shim_other = zp.batchMatPolyZonotope(other.Z.unsqueeze(-2).unsqueeze(0),other.n_dep_gens,other.expMat,other.id,copy_Z=False)
            Z, n_dep_gens, expMat, id = _matmul_genmpz_impl(shim_other, self)
            return zp.batchPolyZonotope(Z.squeeze(-2).squeeze(0), n_dep_gens, expMat, id).compress(2)
        
        elif isinstance(other,matPolyZonotope):
            args = _matmul_genmpz_impl(other, self)
            return batchMatPolyZonotope(*args).compress(2)

        else:
            return NotImplemented

    def to_batchMatZonotope(self):
        if self.n_dep_gens != 0:
            ind = torch.any(self.expMat%2,1)
            Gquad = self.G[self.batch_idx_all+(~ind,)]
            c = self.C + 0.5*torch.sum(Gquad,-3)
            Z = torch.vstack((c.unsqueeze(-3), self.G[self.batch_idx_all+(ind,)],0.5*Gquad,self.Grest))
        else:
            Z = self.Z
        return zp.batchMatZonotope(Z)

    def reduce_indep(self,order,option='girard'):
        # extract dimensions
        N = self.n_rows*self.n_cols
        Q = self.n_indep_gens

        # number of gens kept (N gens will be added back after reudction)
        K = int(N*order-N)
        # check if the order need to be reduced
        if Q > N*order and K >=0:            
            G = self.Grest
            # caculate the length of the gens with a special metric
            len = torch.sum(G**2,(-1,-2)) # NOTE -1
            # determine the smallest gens to remove 
            ind = torch.argsort(len,dim=-1,descending=True).unsqueeze(-1).unsqueeze(-1).repeat((1,)*(self.batch_dim+1)+self.shape)
            ind_rem, ind_red = ind[self.batch_idx_all+(slice(K),)], ind[self.batch_idx_all+(slice(K,None),)]
            # reduce the generators with the reducetion techniques for linear zonotopes
            d = torch.sum(abs(G.gather(-3,ind_red)),-3).reshape(self.batch_shape+(-1,))
            Gbox = torch.diag_embed(d).reshape(self.batch_shape+(-1,3,3))
            # add the reduced gens as new indep gens
            ZRed = torch.cat((self.C.unsqueeze(-3),self.G,G.gather(-3,ind_rem),Gbox),dim=-3)
        else:
            ZRed = self.Z
        n_dg_red = self.n_dep_gens
        if self.n_rows == 1 == self.n_cols and n_dg_red != 1:            
            ZRed = torch.cat((ZRed[self.batch_idx_all+(0,)],ZRed[self.batch_idx_all+(slice(1,n_dg_red+1),)].sum(-3).unsqueeze(-3),ZRed[self.batch_idx_all+(slice(n_dg_red+1,None),)]),dim=-3)
            n_dg_red = 1
        return batchMatPolyZonotope(ZRed,n_dg_red,self.expMat,self.id,copy_Z=False)
    
    @staticmethod
    def from_mpzlist(mpzlist):
        assert len(mpzlist) > 0, "Expected at least 1 element input!"
        # Check type
        assert np.all([isinstance(mpz, matPolyZonotope) for mpz in mpzlist]), "Expected all elements to be of type matPolyZonotope"
        # Validate dimensions match
        n_mpz = len(mpzlist)
        shape = mpzlist[0].shape
        dtype = mpzlist[0].dtype
        device = mpzlist[0].device
        [mpz.shape for mpz in mpzlist].count(shape) == n_mpz, "Expected all elements to have the same shape!"

        # First loop to extract key parts
        all_ids = [None]*n_mpz
        dep_gens = [None]*n_mpz
        all_c = [None]*n_mpz
        n_grest = [None]*n_mpz
        for i, mpz in enumerate(mpzlist):
            all_ids[i] = mpz.id
            dep_gens[i] = mpz.n_dep_gens
            all_c[i] = mpz.C.unsqueeze(0)
            n_grest[i] = mpz.n_indep_gens
        
        # Combine
        all_ids = np.unique(np.concatenate(all_ids, axis=None))
        all_dep_gens = np.sum(dep_gens)
        dep_gens_idxs = np.cumsum([0]+dep_gens)
        n_grest = np.max(n_grest)
        all_c = torch.stack(all_c)

        # Preallocate
        all_G = torch.zeros((n_mpz, all_dep_gens) + shape, dtype=dtype, device=device)
        all_grest = torch.zeros((n_mpz, n_grest) + shape, dtype=dtype, device=device)
        all_expMat = torch.zeros((all_dep_gens, len(all_ids)), dtype=torch.int64, device=device)
        last_expMat_idx = 0

        # expand remaining values
        for mpzid in range(n_mpz):
            # Expand ExpMat (replace any with nonzero to fix order bug!)
            matches = np.nonzero(np.expand_dims(mpzlist[mpzid].id,1) == all_ids)[1]
            end_idx = last_expMat_idx + mpzlist[mpzid].expMat.shape[0]
            all_expMat[last_expMat_idx:end_idx,matches] = mpzlist[mpzid].expMat
            last_expMat_idx = end_idx
        
            # expand out all G matrices
            all_G[mpzid,dep_gens_idxs[mpzid]:dep_gens_idxs[mpzid+1]] = mpzlist[mpzid].G

            # Expand out all grest
            grest = mpzlist[mpzid].Grest
            all_grest[mpzid,:grest.shape[0]] = grest
        
        # Combine, reduce, output.
        Z = torch.concat((all_c, all_G, all_grest), dim=-3)
        out = zp.batchMatPolyZonotope(Z, all_dep_gens, all_expMat, all_ids).compress(2)
        return out
    
    @staticmethod
    def combine_bmpz(bmpzlist, idxs):
        # Takes a list of bpz and respective idxs for them and combines them appropriately
        out_list = np.empty(np.concatenate(idxs, axis=None).max()+1, dtype=object)
        for i,locations in enumerate(idxs):
            out_list[locations] = [bmpzlist[i][j] for j in range(len(locations))]
        return zp.batchMatPolyZonotope.from_pzlist(out_list)
    
    @staticmethod
    def zeros(batch_size, dim1, dim2=None, dtype=None, device=None):
        dim2 = dim1 if dim2 is not None else dim2
        Z = torch.zeros((batch_size, 1, dim1, dim2), dtype=dtype, device=device)
        expMat = torch.empty((0,0),dtype=torch.int64, device=device)
        id = np.empty(0,dtype=np.int64)
        return zp.batchMatPolyZonotope(Z, 0, expMat=expMat, id=id, copy_Z=False)
    
    @staticmethod
    def ones(batch_size, dim1, dim2=None, dtype=None, device=None):
        dim2 = dim1 if dim2 is not None else dim2
        Z = torch.zeros((batch_size, 1, dim1, dim2), dtype=dtype, device=device)
        expMat = torch.empty((0,0),dtype=torch.int64, device=device)
        id = np.empty(0,dtype=np.int64)
        return zp.batchMatPolyZonotope(Z, 0, expMat=expMat, id=id, copy_Z=False)
    
    @staticmethod
    def eye(batch_size, dim, dtype=None, device=None):
        Z = torch.eye(dim, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1, -1)
        expMat = torch.empty((0,0),dtype=torch.int64, device=device)
        id = np.empty(0,dtype=np.int64)
        return zp.batchMatPolyZonotope(Z, 0, expMat=expMat, id=id, copy_Z=False)
    