"""
Define class for matrix polynomial zonotope
Author: Yongseok Kwon
Reference: CORA, Patrick Holme's implementation
"""
from zonopy.contset.polynomial_zonotope.utils import removeRedundantExponentsBatch, mergeExpMatrix
from zonopy.contset.polynomial_zonotope.poly_zono import polyZonotope
import zonopy as zp
import torch
import numpy as np
from ..gen_ops import (
    _add_genpz_impl,
    _add_genzono_num_impl,
    _add_genpz_zono_impl,
    _mul_genpz_impl,
    _mul_genzono_num_impl,
    _matmul_genmpz_impl,
    )
import zonopy.internal as zpi

class batchPolyZonotope:
    r''' Batched 1D polynomial zonotope

    Batched form of the :class:`polyZonotope` class.
    This class is used to represent a batch of polynomial zonotopes over the same domain
    with arbitrary batch dimensions.
    It follows a similar formulation from the :class:`polyZonotope` class as the
    :class:`batchZonotope` class did from :class:`zonotope`.

    This results in a :math:`\mathbf{Z} \in \mathbb{R}^{B_1 \times B_2 \times \cdots \times B_b \times (N+M+1) \times d}` tensor

    Refer to the :class:`polyZonotope` class for more information polynomial zonotops.
    '''
    # NOTE: property for mat pz
    def __init__(self,Z,n_dep_gens=0,expMat=None,id=None,copy_Z=True, dtype=None, device=None):
        r''' Constructor for the batchPolyZonotope class
        
        Args:
            Z (torch.Tensor): The center and generator matrix of the polynomial zonotope.
                The shape of Z should be :math:`(B_1, B_2, \cdots, B_b, N+M+1, d)` where :math:`B_1, B_2, \cdots, B_b` are the batch dimensions,
                :math:`N` is the number of dependent generators, :math:`M` is the number of independent generators, and :math:`d` is the dimension of the zonotope.
            n_dep_gens (int, optional): The number of dependent generators in the polynomial zonotope. Default is 0.
            expMat (torch.Tensor, optional): The exponent matrix of the dependent generators. If ``None``, it will be the identity matrix. Default: None
            id (torch.Tensor, optional): The integer identifiers for the dependent generators. If ``None``, it will be the range of the number of dependent generators. Default: None
            copy_Z (bool, optional): If ``True``, it will copy the input ``Z`` value. Default: ``True``
            dtype (torch.dtype, optional): The data type of the polynomial zonotope. If ``None``, it will be inferred. Default: ``None``
            device (torch.device, optional): The device of the polynomial zonotope. If ``None``, it will be inferred. Default: ``None``

        Raises:
            AssertionError: If the dimension of Z input is not 3 or more.
            AssertionError: If the exponent matrix does not seem to be valid for the given dependent generators or ids.
            AssertionError: If the number of dependent generators does not match the number of ids.
            AssertionError: If the exponent matrix is not a non-negative integer matrix.
        '''
        # If compress=2, it will always copy.

        # Make sure Z is a tensor and shaped right
        if not isinstance(Z, torch.Tensor) and dtype is None:
            dtype = torch.get_default_dtype()
        Z = torch.as_tensor(Z, dtype=dtype, device=device)
        assert len(Z.shape) > 2, f'The dimension of Z input should be either 1 or 2, not {len(Z.shape)}.'

        self.batch_dim = len(Z.shape) - 2
        self.batch_idx_all = tuple([slice(None) for _ in range(self.batch_dim)])

        # Make an expMat and id if not given
        if expMat is None and id is None:
            self.expMat = torch.eye(n_dep_gens,dtype=torch.long,device=Z.device) # if G is EMPTY_TENSOR, it will be EMPTY_TENSOR, size = (0,0)
            self.id = np.arange(self.expMat.shape[1],dtype=int)
            
        # Otherwise make sure expMat is right
        elif expMat is not None:
            #check correctness of user input
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
            nonzero_g = torch.sum(self.G!=0,tuple(range(self.batch_dim))+(-1,))!=0 # non-zero generator index
            G = self.G[...,nonzero_g,:]
            expMat = self.expMat[nonzero_g]

        # Remove generators related to redundant exponents
        elif compression_level == 2: 
            expMat, G = removeRedundantExponentsBatch(self.expMat, self.G, [])

        else:
            raise ValueError("Can only compress to 1 or 2!")

        # Update self
        self.Z = torch.cat((self.c.unsqueeze(-2), G, self.Grest), dim=-2)
        self.expMat = expMat
        self.n_dep_gens = G.shape[-2]

        # For chaining
        return self

    def __getitem__(self,idx):
        Z = self.Z[idx]
        if len(Z.shape) > 2:
            return batchPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False)
        else:
            return polyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False)
    @property 
    def batch_shape(self):
        return self.Z.shape[:-2]
    @property
    def itype(self):
        return self.expMat.dtype
    @property 
    def dtype(self):
        return self.Z.dtype 
    @property
    def device(self):
        return self.Z.device
    @property 
    def c(self):
        return self.Z[self.batch_idx_all+(0,)]
    @property 
    def G(self):
        return self.Z[self.batch_idx_all+(slice(1,self.n_dep_gens+1),)]
    @property 
    def Grest(self):
        return self.Z[self.batch_idx_all+(slice(self.n_dep_gens+1,None),)]
    @property
    def n_generators(self):
        return self.Z.shape[-2]-1
    @property
    def n_indep_gens(self):
        return self.Z.shape[-2]-1-self.n_dep_gens
    @property 
    def dimension(self):
        return self.Z.shape[-1]
    @property 
    def shape(self):
        return self.Z.shape[-1:]
    @property 
    def input_pairs(self):
        id_sorted, order = torch.sort(self.id)
        expMat_sorted = self.expMat[:,order] 
        return self.Z, self.n_dep_gens, expMat_sorted, id_sorted

    def to(self,dtype=None,itype=None,device=None):
        Z = self.Z.to(dtype=dtype,device=device, non_blocking=True)
        expMat = self.expMat.to(dtype=itype,device=device, non_blocking=True)
        # id = self.id.to(device=device)
        return batchPolyZonotope(Z,self.n_dep_gens,expMat,self.id,copy_Z=False)

    def cpu(self):
        Z = self.Z.cpu()
        expMat = self.expMat.cpu()
        # id = self.id.cpu()
        return batchPolyZonotope(Z,self.n_dep_gens,expMat,self.id,copy_Z=False)

    def  __add__(self,other):
        '''
        Overloaded '+' operator for Minkowski sum
        self: <polyZonotope>
        other: <torch.tensor> OR <zonotope> OR <polyZonotope>
        return <polyZonotope>
        '''
        # if other is a vector
        if  isinstance(other, (torch.Tensor, float, int)):
            Z = _add_genzono_num_impl(self, other)
            return batchPolyZonotope(Z, self.n_dep_gens, self.expMat, self.id, copy_Z=False)

        # if other is a polynomial zonotope
        elif isinstance(other, (polyZonotope, batchPolyZonotope)): # exact Plus
            args = _add_genpz_impl(self, other, batch_shape=self.batch_shape)
            return batchPolyZonotope(*args).compress(2)
        
        # if other is a zonotope
        elif isinstance(other, (zp.zonotope, zp.batchZonotope)):
            args = _add_genpz_zono_impl(self, other)
            return batchPolyZonotope(*args, copy_Z=False)

        else:
            return NotImplemented
        
    __radd__ = __add__

    def __sub__(self,other):
        import warnings
        warnings.warn(
            "PZ subtraction as addition of negative is deprecated and will be removed to reduce confusion!",
            DeprecationWarning)
        return self.__add__(-other)
    
    def __rsub__(self,other):
        import warnings
        warnings.warn(
            "PZ subtraction as addition of negative is deprecated and will be removed to reduce confusion!",
            DeprecationWarning)
        return -self.__sub__(other)
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        '''
        Overloaded unary '-' operator for negation
        self: <polyZonotope>
        return <polyZonotope>
        '''
        return batchPolyZonotope(torch.cat((-self.Z[:1+self.n_dep_gens],self.Grest)),self.n_dep_gens,self.expMat, self.id,copy_Z=False)

    def __mul__(self,other):
        # if other is a vector
        if isinstance(other,(torch.Tensor,int,float)):
            Z = _mul_genzono_num_impl(self, other, batch_shape=self.batch_shape)
            return batchPolyZonotope(Z, self.n_dep_gens, self.expMat, self.id, copy_Z=False)

        # if other is a polynomial zonotope or batch polynomial zonotope
        elif isinstance(other,(polyZonotope,batchPolyZonotope)):
            args = _mul_genpz_impl(self, other)
            return batchPolyZonotope(*args).compress(2)
        
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a matrix or an interval matrix with a polyZonotope
        self: <polyZonotope>
        other: <torch.tensor> OR <intervals>
        return <polyZonotope>
        '''
        
        # if other is a matrix
        if isinstance(other, torch.Tensor):            
            Z = self.Z@other.transpose(-2,-1)
            return batchPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False).compress(1) # TODO IS THIS RIGHT?
        
        if isinstance(other, zp.matPolyZonotope):
            # Shim self to batchMatPolyZono and return that matmul
            shim_self = zp.batchMatPolyZonotope(self.Z.unsqueeze(-1),self.n_dep_gens,self.expMat,self.id,copy_Z=False)
            Z, n_dep_gens, expMat, id = _matmul_genmpz_impl(other, shim_self)
            return zp.batchPolyZonotope(Z.squeeze(-1), n_dep_gens, expMat, id).compress(2)
        
        else:
            return NotImplemented
        
    # def __len__(self):
    #     return self.Z.shape[0]

    # NOTE - this is a shim for reducing each individual pz in the batch
    def reduce(self, order, option='girard'):
        # Reduce by iterating over all
        # Shim to batchPolyZonotope
        len_ents = int(np.prod(self.batch_shape))
        idx_tuple = np.unravel_index(np.arange(len_ents), self.batch_shape)
        pzlist = [None]*len_ents
        for out_i, idxs in enumerate(zip(*idx_tuple)):
            pzlist[out_i] = self[idxs].reduce(order, option=option)
        return zp.batchPolyZonotope.from_pzlist(pzlist, batch_shape=self.batch_shape)
    
    # TODO Inspect for speedup?
    def reduce_indep(self,order,option='girard'):
        # extract dimensions
        N = self.dimension
        Q = self.n_indep_gens
            
        # number of gens kept (N gens will be added back after reudction)
        K = int(N*order-N)
        # check if the order need to be reduced
        if Q > N*order and K >=0:
            G = self.Grest
            # caculate the length of the gens with a special metric
            len = torch.sum(G**2,-1) # NOTE -1
            # determine the smallest gens to remove            
            ind = torch.argsort(len,dim=-1,descending=True).unsqueeze(-1).repeat((1,)*(self.batch_dim+1)+self.shape)
            ind_rem, ind_red = ind[self.batch_idx_all+(slice(K),)], ind[self.batch_idx_all+(slice(K,None),)]
            # reduce the generators with the reducetion techniques for linear zonotopes
            d = torch.sum(abs(G.gather(-2,ind_red)),-2)
            Gbox = torch.diag_embed(d)
            # add the reduced gens as new indep gens
            ZRed = torch.cat((self.c.unsqueeze(-2),self.G,G.gather(-2,ind_rem),Gbox),dim=-2)
        else:
            ZRed = self.Z
        n_dg_red = self.n_dep_gens
        if self.dimension == 1 and n_dg_red != 1:            
            ZRed = torch.cat((ZRed[self.batch_idx_all+(0,)],ZRed[self.batch_idx_all+(slice(1,n_dg_red+1),)].sum(-2).unsqueeze(-2),ZRed[self.batch_idx_all+(slice(n_dg_red+1,None),)]),dim=-2)
            n_dg_red = 1
        return batchPolyZonotope(ZRed,n_dg_red,self.expMat,self.id,copy_Z=False)

    def exactCartProd(self,other):
        '''
        self: <polyZonotope>
        other: <polyZonotope>
        return <polyZonotope>
        '''

        # Get the relevant center
        if isinstance(other,polyZonotope):
            c = torch.hstack([self.c,other.c.expand(self.c[:-1]+(other.dimension,))]).unsqueeze(-2)
        elif isinstance(other,batchPolyZonotope):
            c = torch.hstack((self.c,other.c)).unsqueeze(-2)
        else:
            raise TypeError
        
        # Update the expMat
        id,expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
        expMat = torch.vstack((expMat1,expMat2))

        # Construct the new G
        new_G_shape = self.G.shape[:-2] + (self.G.shape[-2] + other.G.shape[-2], self.G.shape[-1] + other.G.shape[-1])
        g1_slice = self.batch_idx_all + (slice(None,self.G.shape[-2]),slice(None,self.G.shape[-1]))
        g2_slice = self.batch_idx_all + (slice(self.G.shape[-2],None),slice(self.G.shape[-1],None))
        G = torch.zeros(new_G_shape, dtype=self.dtype, device=self.device)
        G[g1_slice] = self.G
        G[g2_slice] = other.G

        # Construct the new Grest
        new_Grest_shape = self.Grest.shape[:-2] + (self.Grest.shape[-2] + other.Grest.shape[-2], self.Grest.shape[-1] + other.Grest.shape[-1])
        g1_slice = self.batch_idx_all + (slice(None,self.Grest.shape[-2]),slice(None,self.Grest.shape[-1]))
        g2_slice = self.batch_idx_all + (slice(self.Grest.shape[-2],None),slice(self.Grest.shape[-1],None))
        Grest = torch.zeros(new_Grest_shape, dtype=self.dtype, device=self.device)
        Grest[g1_slice] = self.Grest
        Grest[g2_slice] = other.Grest

        Z = torch.cat((c,G,Grest),dim=-2)
        n_dep_gens = self.n_dep_gens+other.n_dep_gens
        return batchPolyZonotope(Z,n_dep_gens,expMat,id).compress(2)

    def to_batchZonotope(self):
        if self.n_dep_gens != 0:
            ind = torch.any(self.expMat%2,1)
            Gquad = self.G[self.batch_idx_all+(~ind,)]
            c = self.c + 0.5*torch.sum(Gquad,-2)
            Z = torch.cat((c.unsqueeze(-2), self.G[self.batch_idx_all+(ind,)],0.5*Gquad,self.Grest),-2)
        else: 
            Z = self.Z
        return zp.batchZonotope(Z)

    def to_interval(self,method='interval'):
        if method == 'interval':
            return self.to_batchZonotope().to_interval()
        else:
            assert False, 'Not implemented'

    # TODO Inspect for speedup?
    def slice_dep(self,id_slc,val_slc):
        '''
        Slice polynomial zonotpe in depdent generators
        id_slc: id to dlice
        val_slc: indeterminant to slice
        '''
        if isinstance(id_slc,(int,list)):
            if isinstance(id_slc,int):
                id_slc = [id_slc]
            id_slc = np.array(id_slc)
        if isinstance(val_slc,(int,float,list)):
            if isinstance(val_slc,(int,float)):
                val_slc = [val_slc]
            val_slc = torch.tensor(val_slc,dtype=self.dtype,device=self.device)
        
        if any(abs(val_slc)>1):
            import pdb; pdb.set_trace()
        #assert all(val_slc<=1) and all(val_slc>=-1), 'Indereminant should be in [-1,1].'
        
        id_slc, val_slc = id_slc.reshape(-1,1), val_slc.reshape(1,-1)
        order = np.argsort(id_slc.reshape(-1))
        id_slc, val_slc  = id_slc[order], val_slc[:,order]
        ind = np.any(self.id==id_slc,dim=0) # corresponding id for self.id  
        ind2 = np.any(self.id==id_slc,dim=1)# corresponding id for id_slc
        #assert ind.numel()==len(id_slc), 'Some specidied IDs do not exist!'
        if ind.shape[0] != 0:
            G = self.G*torch.prod(val_slc[:,ind2]**self.expMat[:,ind],dim=1)
            expMat = self.expMat[:,~ind]
            id = self.id[:,~ind]
        else:
            G = self.G
            expMat = self.expMat
            id = self.id

        #expMat, G = removeRedundantExponents(expMat,G)
        ind = torch.sum(expMat,1) == 0
        if torch.any(ind):
            c = self.c + torch.sum(G[ind],0)
            G = G[~ind]
            expMat = expMat[~ind]
        else:
            c = self.c
        '''
        id = self.id
        ind = torch.sum(expMat,0) == 0
        if torch.any(ind):
            expMat = expMat[:,~ind]
            id = id[:,~ind]
        '''
        
        if G.shape[0] == 0 and self.Grest.shape[0] == 0:
            return polyZonotope(c,0,expMat,id).compress(2)
        else:
            return polyZonotope(torch.vstack((c,G,self.Grest)), G.shape[0],expMat,id).compress(2)
    

    def center_slice_all_dep(self,val_slc):
        # get all values in order
        val_slc = val_slc[..., None, self.id] # Batch dims, ..., 1, n_ids
        # Exponentiate by exponent matrix, reduce the product for each term, then multiply by each dep gen
        # offset = torch.prod(val_slc**self.expMat, dim=-1).unsqueeze(-2)@self.G
        # Torch einsum accomplishes the above with better accuracy and arbitrary dimensions
        alpha_coeffs = torch.prod(val_slc**self.expMat, dim=-1)
        offset = torch.einsum('...g,...gd->...d',
                              alpha_coeffs,
                              self.G) # b1, b2,..., dim
        return self.c + offset.squeeze(-2)

    def grad_center_slice_all_dep(self,val_slc):
        # prepare output
        grad = torch.zeros(self.batch_shape + self.shape + val_slc.shape[-1:], dtype=self.dtype, device=self.device)
        n_ids = len(self.id)

        # get all values in order
        val_slc = val_slc[..., None, None, self.id] # Batch dims, ..., 1, 1, n_ids
        expMat_red = self.expMat.expand(n_ids, -1, -1) - torch.eye(n_ids, dtype=self.expMat.dtype, device=self.device).unsqueeze(-2) # a tensor of reduced order expMat for each column, n_ids,  n_dep_gens, n_ids
        # grad[..., self.id] = ((self.expMat.T*torch.prod(val_slc**expMat_red,dim=-1).nan_to_num())@self.G).transpose(-1,-2) # b1, b2,..., dim, n_ids
        # Torch einsum accomplishes the above with better accuracy and arbitrary dimensions
        alpha_coeffs = self.expMat.T*torch.prod(val_slc**expMat_red,dim=-1).nan_to_num()
        # grad[..., self.id] = torch.einsum('...ig,...gd->...di',
        #                                   alpha_coeffs,
        #                                   self.G) # b1, b2,..., dim, n_ids
        # Actually, in this case, manually writing it is faster
        grad[..., self.id] = (alpha_coeffs@self.G).transpose(-1,-2)
        return grad

    # TODO Unverified since update
    def hess_center_slice_all_dep(self,val_slc):
        n_ids= self.id.shape[0]
        n_vals = val_slc.shape[-1]
        val_slc = val_slc[self.batch_idx_all + (slice(n_ids),)].reshape(self.batch_shape+(1,1,1,n_ids)) # b1, b2,..., 1, 1, 1, n_ids
        expMat = self.expMat[:,np.argsort(self.id)] # n_dep_gens, n_ids
        expMat_red = expMat.unsqueeze(0).repeat(n_ids,1,1) - torch.eye(n_ids,dtype=int).unsqueeze(-2) # a tensor of reduced order expMat for each column
        expMat_twice_red = expMat.reshape((1,1)+expMat.shape).repeat(n_ids,n_ids,1,1) - torch.eye(n_ids,dtype=int).unsqueeze(-2) - torch.eye(n_ids,dtype=int).reshape(n_ids,1,1,n_ids)
        expMat_first = expMat.T.unsqueeze(1).repeat(1,n_ids,1)
        hess = torch.zeros(self.batch_shape+(self.dimension,n_vals,n_vals),dtype=self.dtype,device=self.device)
        hess[self.batch_idx_all+(slice(None),slice(n_ids),slice(n_ids))] = ((expMat_first*expMat_red.transpose(-1,-2)*torch.prod(val_slc**expMat_twice_red,dim=-1).nan_to_num())@self.G.unsqueeze(-3)).transpose(-3,-1)
        return hess


    def slice_all_dep(self,val_slc):
        '''
        Slice polynomial zonotpe in depdent generators
        id_slc: id to slice
        val_slc: indeterminant to slice
        c: <torch.Tensor>, shape [b1,b2,...,nx]
        grad_c: <torch.Tensor>, d c/d val_slc
        shape [b1,b2,...,nx,n_ids]
        '''

        ##################################
        return zp.batchZonotope(torch.cat((self.center_slice_all_dep(val_slc).unsqueeze(-2),self.Grest), -2))


    def deleteZerosGenerators(self,eps=0):
        expMat, G = removeRedundantExponentsBatch(expMat,G,self.batch_idx_all)
        ind = torch.sum(expMat,1) == 0
        if torch.any(ind):
            c = self.c + torch.sum(G[ind],0)
            G = G[~ind]
            expMat = expMat[~ind]
        else:
            c = self.c
        
        id = self.id
        ind = (torch.sum(expMat,0) == 0).cpu().numpy()
        if np.any(ind):
            expMat = expMat[:,~ind]
            id = id[:,~ind]
        return polyZonotope(torch.vstack((c,G,self.Grest)),G.shape[0],expMat,id,copy_Z=False).compress(2)

    def project(self,dim=[0,1]):
        Z = self.Z[self.batch_idx_all+(slice(None),dim)]
        return batchPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,copy_Z=False).compress(1)
    
    def split_dep_indep(self, center_on_dep=True):
        Z_dep = torch.clone(self.Z[...,:self.n_dep_gens+1,:])
        Z_indep = torch.clone(self.Z[...,-(self.n_indep_gens+1):,:])
        Z_indep[...,0,:] *= 0
        if not center_on_dep:
            Z_indep[...,0,:] += Z_dep[...,0,:]
            Z_dep[...,0,:] += 0
        deps = batchPolyZonotope(Z_dep,self.n_dep_gens,self.expMat,self.id,copy_Z=False)
        indeps = zp.batchZonotope(Z_indep)
        return deps, indeps

    @staticmethod
    def from_pzlist(pzlist, batch_shape=None):
        assert len(pzlist) > 0, "Expected at least 1 element input!"
        # Check type
        assert np.all([isinstance(pz, polyZonotope) for pz in pzlist]), "Expected all elements to be of type polyZonotope"
        # Validate dimensions match
        n_pz = len(pzlist)
        dim = pzlist[0].dimension
        dtype = pzlist[0].dtype
        device = pzlist[0].device
        [pz.dimension for pz in pzlist].count(dim) == n_pz, "Expected all elements to have the same dimensions!"

        # First loop to extract key parts
        all_ids = [None]*n_pz
        dep_gens = [None]*n_pz
        all_c = [None]*n_pz
        n_grest = [None]*n_pz
        for i, pz in enumerate(pzlist):
            all_ids[i] = pz.id
            dep_gens[i] = pz.n_dep_gens
            all_c[i] = pz.c.unsqueeze(0)
            n_grest[i] = pz.n_indep_gens
        
        # Combine
        all_ids = np.unique(np.concatenate(all_ids, axis=None))
        all_dep_gens = np.sum(dep_gens)
        dep_gens_idxs = np.cumsum([0]+dep_gens)
        n_grest = np.max(n_grest)
        all_c = torch.stack(all_c)

        # Preallocate
        all_G = torch.zeros((n_pz, all_dep_gens, dim), dtype=dtype, device=device)
        all_grest = torch.zeros((n_pz, n_grest, dim), dtype=dtype, device=device)
        all_expMat = torch.zeros((all_dep_gens, len(all_ids)), dtype=torch.int64, device=device)
        last_expMat_idx = 0

        # expand remaining values
        for pzid in range(n_pz):
            # Expand ExpMat (replace any with nonzero to fix order bug!)
            matches = np.nonzero(np.expand_dims(pzlist[pzid].id,1) == all_ids)[1]
            end_idx = last_expMat_idx + pzlist[pzid].expMat.shape[0]
            all_expMat[last_expMat_idx:end_idx,matches] = pzlist[pzid].expMat
            last_expMat_idx = end_idx
        
            # expand out all G matrices
            all_G[pzid,dep_gens_idxs[pzid]:dep_gens_idxs[pzid+1]] = pzlist[pzid].G

            # Expand out all grest
            grest = pzlist[pzid].Grest
            all_grest[pzid,:grest.shape[0]] = grest
        
        # Combine, reduce, output.
        Z = torch.concat((all_c, all_G, all_grest), dim=-2)
        if batch_shape is not None:
            Z = Z.reshape(batch_shape + Z.shape[-2:])
        out = zp.batchPolyZonotope(Z, all_dep_gens, all_expMat, all_ids, copy_Z=False).compress(2)
        return out
    
    @staticmethod
    def combine_bpz(bpzlist, idxs):
        # Takes a list of bpz and respective idxs for them and combines them appropriately
        out_list = np.empty(np.concatenate(idxs, axis=None).max()+1, dtype=object)
        for i,locations in enumerate(idxs):
            out_list[locations] = [bpzlist[i][j] for j in range(len(locations))]
        return zp.batchPolyZonotope.from_pzlist(out_list)
    
    @staticmethod
    def zeros(batch_size, dims, dtype=None, device=None):
        if not isinstance(batch_size, tuple):
            batch_size = (batch_size,)
        Z = torch.zeros((1, dims), dtype=dtype, device=device).expand(*batch_size, -1, -1)
        expMat = torch.empty((0,0),dtype=torch.int64, device=device)
        id = np.empty(0,dtype=np.int64)
        return zp.batchPolyZonotope(Z, 0, expMat=expMat, id=id, copy_Z=False)
    
    @staticmethod
    def ones(batch_size, dims, dtype=None, device=None):
        if not isinstance(batch_size, tuple):
            batch_size = (batch_size,)
        Z = torch.ones((1, dims), dtype=dtype, device=device).expand(*batch_size, -1, -1)
        expMat = torch.empty((0,0),dtype=torch.int64, device=device)
        id = np.empty(0,dtype=np.int64)
        return zp.batchPolyZonotope(Z, 0, expMat=expMat, id=id, copy_Z=False)