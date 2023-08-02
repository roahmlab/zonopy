"""
Define class for matrix polynomial zonotope
Author: Yongseok Kwon
Reference: CORA, Patrick Holme's implementation
"""
from zonopy.conSet.polynomial_zonotope.utils import removeRedundantExponents, mergeExpMatrix, pz_repr
# from zonopy.conSet import PROPERTY_ID
import zonopy as zp
import torch
import numpy as np
from .gen_ops import (
    _add_genpz_impl,
    _add_genpz_num_impl,
    _add_genpz_zono_impl,
    _mul_genpz_impl,
    _mul_genpz_num_impl,
    )
import zonopy.internal as zpi

class polyZonotope:
    '''
    pZ: <polyZonotope>
    
    Z: <torch.Tensor> center vector and generators matrix Z = [[c],[G],[Grest]]
    , shape [B1, B2, .. , Bb, N+M+1, nx]
    c: <torch.Tensor> center vector of the polyonmial zonotope
    , shape: [nx] 
    G: <torch.Tensor> generator matrix containing the dependent generators
    , shape: [N, nx]
    Grest: <torch.Tensor> generator matrix containing the independent generators
    , shape: [M, nx]
    expMat: <troch.Tensor> matrix containing the exponents for the dependent generators
    , shape: [N, p]
    id: <torch.Tensor> vector containing the integer identifiers for the dependent factors
    , shape: [p]
    compress: <int> level for compress operation on dependent generators with exponent matrix
    , 0: no compress, 1: remove zero dependent generators, 2: remove zero dependent generators and remove redundant expodent

    Eq. 
    G = [[gd1],[gd2],...,[gdN]]
    Grest = [[gi1],[gi2],...,[giM]]
    expMat = [[i11,i12,...,i1N],[i21,i22,...,i2N],...,[ip1,ip2,...,ipN]]
    id = [0,1,2,...,p-1]

    pZ = {
        c + a1*gi1 + a2*gi2 + ... + aN*giN + b1^i11*b2^i21*...*bp^ip1*gd1 + b1^i12*b2^i22*...*bp^ip2*gd2 + ...
        + b1^i1M*b2^i2M*...*bp^ipM*gdM
        | coeff. a1,a2,...,aN; b1,b2,...,bp \in [0,1]
    }
    '''
    def __init__(self,Z,n_dep_gens=0,expMat=None,id=None,compress=2,copy_Z=True):
        # If compress=2, it will always copy.

        # Make sure Z is a tensor
        if not isinstance(Z, torch.Tensor):
            Z = torch.as_tensor(Z,dtype=torch.float)
        
        # Isolate G and the indices
        G_ind = np.arange(1, 1+n_dep_gens)
        G = Z[slice(1, 1+n_dep_gens)]
        
        # Remove zero generators
        if compress == 1:
            nonzero_g = (torch.sum(G!=0,-1)!=0).cpu().numpy() # non-zero generator index
            G_ind = G_ind[nonzero_g]
            G = Z[G_ind]

        # Make an expMat and id if not given
        if expMat is None and id is None:
            self.expMat = torch.eye(G_ind.shape[0],dtype=torch.long,device=Z.device) # if G is EMPTY_TENSOR, it will be EMPTY_TENSOR, size = (0,0)Z
            self.id = np.arange(self.expMat.shape[1],dtype=int)
            
        # Otherwise make sure expMat is right
        elif expMat is not None:
            if not isinstance(expMat, torch.Tensor):
                expMat = torch.as_tensor(expMat,dtype=torch.long,device=Z.device)
            assert expMat.shape[0] == n_dep_gens, 'Invalid exponent matrix.' 
            if zpi.__debug_extra__: assert torch.all(expMat >= 0), 'Invalid exponent matrix.' 
            
            # Remove generators related to redundant exponents
            if compress == 2: 
                self.expMat,G = removeRedundantExponents(expMat,G)
                copy_Z = True
                
            # Remove relevant zero generator indices
            elif compress == 1:
                self.expMat = expMat[nonzero_g]
                
            # Everything is fine
            else:
                self.expMat = expMat

            # Make sure ID is right
            if id is not None:
                self.id = np.asarray(id, dtype=int).flatten()
            else:
                self.id = np.arange(self.expMat.shape[1],dtype=int)
        
        # Otherwise ID is given, but not the expMat, so make identity
        else:
            self.id = np.array(id, dtype=int).flatten()
            assert len(self.id) == n_dep_gens, 'Number of dependent generators must match number of id\'s!'
            self.expMat = torch.eye(G_ind.shape[0],dtype=torch.long,device=Z.device)


        # Copy the Z if requested
        if copy_Z:
            self.Z = torch.vstack((Z[0], G, Z[1+n_dep_gens:]))
        # Otherwise save view if needed
        elif compress == 1:
            Grest_ind = np.arange(1+n_dep_gens, Z.shape[0])
            ind = np.concatenate([[0], G_ind, Grest_ind])
            self.Z = Z[ind]
        # Or save it itself
        else:
            self.Z = Z
            
        # Update n_dep_gens
        self.n_dep_gens = G.shape[0]

    @property
    def itype(self):
        '''
        The data type of a polynomial zonotope exponent matrix
        return torch.short, torch.int, torch.long
        '''
        return self.expMat.dtype
    @property 
    def dtype(self):
        '''
        The data type of vector elements (ex. center) of a polynomial zonotope
        return torch.float or torch.double
        '''
        return self.Z.dtype 
    @property
    def device(self):
        '''
        The device of a polynomial zonotope properties
        return 'cpu', 'cuda:0', or ...
        '''
        return self.Z.device
    @property 
    def c(self):
        '''
        The center of a polynimal zonotope
        return <torch.Tensor>
        , shape [nx]
        '''
        return self.Z[0]
    @property 
    def G(self):
        '''
        Dependent generators of a polynimal zonotope
        return <torch.Tensor>
        , shape [N, nx]
        '''
        return self.Z[1:self.n_dep_gens+1]
    @property
    def Grest(self):
        '''
        Independent generators of a polynimal zonotope
        return <torch.Tensor>
        , shape [M, nx]
        '''
        return self.Z[self.n_dep_gens+1:]
    @property
    def n_generators(self):
        return len(self.Z)-1
    @property
    def n_indep_gens(self):
        return len(self.Z)-1-self.n_dep_gens
    @property 
    def dimension(self):
        return self.Z.shape[1]
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
        return polyZonotope(Z,self.n_dep_gens,expMat,self.id,compress=0,copy_Z=False)
    def cpu(self):
        Z = self.Z.cpu()
        expMat = self.expMat.cpu()
        # id = self.id.cpu()
        return polyZonotope(Z,self.n_dep_gens,expMat,self.id,compress=0,copy_Z=False)


    def __str__(self):
        if self.expMat.numel() == 0:
            expMat_print = torch.tensor([])
        else:
            expMat_print = self.expMat[:,np.argsort(self.id)]
        
        pz_str = f"""center: \n{self.c.to(dtype=torch.float)} \n\nnumber of dependent generators: {self.G.shape[-1]} 
            \ndependent generators: \n{self.G.to(dtype=torch.float)}  \n\nexponent matrix: \n {expMat_print.to(dtype=torch.long)}
            \nnumber of independent generators: {self.Grest.shape[-1]} \n\nindependent generators: \n {self.Grest.to(dtype=torch.float)}
            \ndimension: {self.dimension} \ndtype: {self.dtype}\nitype: {self.itype}\ndtype: {self.device}"""
        
        del_dict = {'tensor':' ','    ':' ','(':'',')':''}
        for del_el in del_dict.keys():
            pz_str = pz_str.replace(del_el,del_dict[del_el])
        return pz_str
 
    def __repr__(self):
        return pz_repr(self)

    def  __add__(self,other):
        '''
        Overloaded '+' operator for Minkowski sum
        self: <polyZonotope>
        other: <torch.tensor> OR <zonotope> OR <polyZonotope>
        return <polyZonotope>
        '''
        # if other is a polynomial zonotope
        if isinstance(other, polyZonotope): # exact Plus
            args = _add_genpz_impl(self, other)
            return polyZonotope(*args)
        
        # if other is a vector
        elif isinstance(other, (torch.Tensor, float, int)):
            args = _add_genpz_num_impl(self, other)
            return polyZonotope(*args, compress=0, copy_Z=False)
        
        # if other is a zonotope
        elif isinstance(other, zp.zonotope):
            args = _add_genpz_zono_impl(self, other)
            return polyZonotope(*args, compress=0, copy_Z=False)
        
        else:
            return NotImplemented
    
    __radd__ = __add__

    def __sub__(self,other):
        return self.__add__(-other)
    
    def __rsub__(self,other):
        return -self.__sub__(other)
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        '''
        Overloaded unary '-' operator for negation
        self: <polyZonotope>
        return <polyZonotope>
        '''
        return polyZonotope(torch.vstack((-self.Z[:1+self.n_dep_gens],self.Grest)),self.n_dep_gens,self.expMat, self.id,compress=0,copy_Z=False)

    def __mul__(self,other):
        # if other is a vector
        if isinstance(other,(torch.Tensor,int,float)):
            args = _mul_genpz_num_impl(self, other)
            return polyZonotope(*args, compress=1, copy_Z=False)
        
        # if other is a polynomial zonotope
        elif isinstance(other,polyZonotope):
            args = _mul_genpz_impl(self, other)
            return polyZonotope(*args)
        
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
            Z = self.Z@other.T
            return polyZonotope(Z,self.n_dep_gens,self.expMat,self.id,compress=1,copy_Z=False)
        else:
            return NotImplemented

    def reduce(self,order,option='girard'):
        # extract dimensions
        N = self.dimension
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
            len = torch.sum(G**2,1)
            # determine the smallest gens to remove            
            ind = torch.argsort(len,descending=True)
            ind_rem,ind_red = ind[:K], ind[K:]
            # split the indices into the ones for dependent and independent
            indDep_red = ind_red[ind_red< P]
            ind_RED = torch.hstack((indDep_red, ind_red[ind_red >= P]))

            indDep_rem = ind_rem[ind_rem < P]
            ind_REM = torch.hstack((indDep_rem,ind_rem[ind_rem >= P]))
            # construct a zonotope from the gens that are removed
            n_dg_red = indDep_red.shape[0]
            Ered = self.expMat[indDep_red]
            Ztemp = torch.vstack((torch.zeros(N,dtype=self.dtype,device=self.device),G[ind_RED]))
            pZtemp = polyZonotope(Ztemp,n_dg_red,Ered,self.id,compress=1) # NOTE: ID???
            zono = pZtemp.to_zonotope() # zonotope over-approximation
            # reduce the constructed zonotope with the reducetion techniques for linear zonotopes
            zonoRem = zono.reduce(1,option)
            
            # remove the gens that got reduce from the gen matrices
            expMatRem = self.expMat[indDep_rem]  
            n_dg_rem = indDep_rem.shape[0]
            # add the reduced gens as new indep gens
            ZRed = torch.vstack((self.c + zonoRem.center,G[ind_REM],zonoRem.generators))
        else:
            ZRed = self.Z
            n_dg_rem = self.n_dep_gens
            expMatRem = self.expMat
        # remove all exponent vector dimensions that have no entries
        ind = (torch.sum(expMatRem,0)>0).cpu().numpy()
        #ind = temp.nonzero().reshape(-1)
        expMatRem = expMatRem[:,ind]
        idRem = self.id[ind]
        if self.dimension == 1:
            ZRed = torch.vstack((ZRed[0],ZRed[1:n_dg_red+1].sum(0),ZRed[n_dg_red+1:]))
            n_dg_rem = 1
        return polyZonotope(ZRed,n_dg_rem,expMatRem,idRem,compress=1,copy_Z=False)

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
            len = torch.sum(G**2,1)
            # determine the smallest gens to remove            
            ind = torch.argsort(len,descending=True)
            ind_rem,ind_red = ind[:K], ind[K:]
            # reduce the generators with the reducetion techniques for linear zonotopes
            d = torch.sum(abs(G[ind_red]),0)
            Gbox = torch.diag(d)
            # add the reduced gens as new indep gens
            ZRed = torch.vstack((self.c,self.G,G[ind_rem],Gbox))
        else:
            ZRed = self.Z
        n_dg_red = self.n_dep_gens
        if self.dimension == 1 and n_dg_red != 1:
            ZRed = torch.vstack((ZRed[0],ZRed[1:n_dg_red+1].sum(0),ZRed[n_dg_red+1:]))
            n_dg_red = 1
        return polyZonotope(ZRed,n_dg_red,self.expMat,self.id,compress=1,copy_Z=False)

    def exactCartProd(self,other):
        '''
        self: <polyZonotope>
        other: <polyZonotope>
        return <polyZonotope>
        '''    
        if isinstance(other,polyZonotope):
            c = torch.hstack((self.c,other.c))
            id,expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            G = torch.block_diag(self.G,other.G)
            expMat = torch.vstack((expMat1,expMat2))
            Grest = torch.block_diag(self.Grest,other.Grest)
        Z = torch.vstack((c,G,Grest))
        n_dep_gens = self.n_dep_gens+other.n_dep_gens
        return polyZonotope(Z,n_dep_gens,expMat,id)

    def to_zonotope(self):
        if self.n_dep_gens !=0:
            ind = torch.any(self.expMat%2,1)
            Gquad = self.G[~ind]
            c = self.c + 0.5*torch.sum(Gquad,0)
            Z = torch.vstack((c,self.G[ind],0.5*Gquad,self.Grest))
        else:
            Z = self.Z
        return zp.zonotope(Z)

    def to_interval(self,method='interval'):
        if method == 'interval':
            return self.to_zonotope().to_interval()
        else:
            assert False, 'Not implemented'

    def slice_dep(self,id_slc,val_slc):
        '''
        Slice polynomial zonotpe in depdent generators
        id_slc: id to slice
        val_slc: indeterminant to slice
        '''
        if isinstance(id_slc,(int,list)):
            if isinstance(id_slc,int):
                id_slc = [id_slc]
            # id_slc = torch.tensor(id_slc,dtype=self.dtype,device=self.device)
            id_slc = np.array(id_slc, dtype=int)
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
            return polyZonotope(c,0,expMat,id)
        else:
            return polyZonotope(torch.vstack((c,G,self.Grest)), G.shape[0],expMat,id)

    def center_slice_all_dep(self,val_slc):
        n_ids= self.id.shape[0]
        val_slc = val_slc[:n_ids]
        expMat = self.expMat[:,np.argsort(self.id)]
        return self.c + torch.sum(self.G*torch.prod(val_slc**expMat,dim=-1).unsqueeze(-1),0)

    def grad_center_slice_all_dep(self,val_slc):
        n_ids= self.id.shape[0]
        val_slc = val_slc[:n_ids]
        expMat = self.expMat[:,np.argsort(self.id)]
        expMat_red = expMat.unsqueeze(0).repeat(n_ids,1,1) - torch.eye(n_ids).unsqueeze(-2) # a tensor of reduced order expMat for each column
        return (self.G*(expMat.T*torch.prod(val_slc**expMat_red,dim=-1).nan_to_num()).unsqueeze(-1)).sum(1).T
        # return (expMat.T*torch.prod(val_slc**expMat_red,dim=-1).nan_to_num()@self.G).T

    def hess_center_slice_all_dep(self,val_slc):
        n_ids= self.id.shape[0]
        val_slc = val_slc[:n_ids]
        expMat = self.expMat[:,np.argsort(self.id)]
        expMat_red = expMat.unsqueeze(0).repeat(n_ids,1,1) - torch.eye(n_ids,dtype=int).unsqueeze(-2) # a tensor of reduced order expMat for each column
        expMat_twice_red = expMat.reshape((1,1)+expMat.shape).repeat(n_ids,n_ids,1,1) - torch.eye(n_ids,dtype=int).unsqueeze(-2) - torch.eye(n_ids,dtype=int).reshape(n_ids,1,1,n_ids)
        expMat_first = expMat.T.unsqueeze(1).repeat(1,n_ids,1)
        return (self.G*(expMat_first*expMat_red.transpose(-1,-2)*torch.prod(val_slc**expMat_twice_red,dim=-1).nan_to_num()).unsqueeze(-1)).sum(-2).squeeze(-1).transpose(0,-1)

    def slice_all_dep(self,val_slc):
        '''
        Slice polynomial zonotpe in all depdent generators

        
        id_slc: id to slice
        val_slc: indeterminant to slice
        return,
        c: <torch.Tensor>, shape [nx]
        grad_c: <torch.Tensor>, shape [n_ids,nx]

        '''

        ##################################
        n_ids= self.id.shape[0]
        val_slc = val_slc[:n_ids]
        expMat = self.expMat[:,np.argsort(self.id)]
        c = self.c + torch.sum(self.G*torch.prod(val_slc**expMat,dim=-1).unsqueeze(-1),0) 
        return zp.zonotope(torch.vstack((c,self.Grest)))



    def deleteZerosGenerators(self,eps=0):
        expMat, G = removeRedundantExponents(self.expMat,self.G,eps)
        ind = torch.sum(expMat,1) == 0
        if torch.any(ind):
            c = self.c + torch.sum(G[ind],0)
            G = G[~ind]
            expMat = expMat[~ind]
        else:
            c = self.c
        
        id = self.id
        ind = torch.sum(expMat,0) == 0
        if torch.any(ind):
            expMat = expMat[:,~ind]
            id = id[:,~ind]
        return polyZonotope(torch.vstack((c,G,self.Grest)),G.shape[0],expMat,id,compress=0,copy_Z=False)


    def project(self,dim=[0,1]):
        return polyZonotope(self.Z[:,dim],self.n_dep_gens,self.expMat,self.id,compress=1,copy_Z=False)
    '''
    def plot(self,dim=[0,1]):
        pz = self.project(dim)
    '''
    
    @staticmethod
    def zeros(dims):
        Z = torch.zeros((1, dims))
        expMat = torch.empty((0,0),dtype=torch.int64)
        id = np.empty(0,dtype=np.int64)
        return zp.polyZonotope(Z, 0, expMat=expMat, id=id, compress=0, copy_Z=False)
    
    @staticmethod
    def ones(dims):
        Z = torch.ones((1, dims))
        expMat = torch.empty((0,0),dtype=torch.int64)
        id = np.empty(0,dtype=np.int64)
        return zp.polyZonotope(Z, 0, expMat=expMat, id=id, compress=0, copy_Z=False)

if __name__ == '__main__':
    #pz = polyZonotope(torch.tensor([1.212,24142.42]),torch.eye(2),torch.eye(2),dtype=float,itype=int)
    #print(pz.__repr__())

    pz = polyZonotope(torch.tensor([[1]]),0)
    import pdb;pdb.set_trace()
    #print(pz)
    
    #print(pz.reduce(10))







        