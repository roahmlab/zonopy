"""
Define class for matrix polynomial zonotope
Author: Yongseok Kwon
Reference: Patrick Holme's implementation
"""
from zonopy.conSet.polynomial_zonotope.utils import removeRedundantExponentsBatch, mergeExpMatrix
from zonopy.conSet import PROPERTY_ID
from zonopy import batchPolyZonotope
from zonopy import matPolyZonotope
import zonopy as zp
import torch

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
    def __init__(self,Z,n_dep_gens=0,expMat=None,id=None,prop='None'):
        assert isinstance(prop,str), 'Property should be string.'
        assert isinstance(Z, torch.Tensor), 'The input matrix should be either torch tensor or list.'
        assert len(Z.shape) > 3, f'The dimension of Z input should be either 1 or 2, not {len(Z.shape)}.'
        self.batch_dim = len(Z.shape) - 3
        self.batch_idx_all = tuple([slice(None) for _ in range(self.batch_dim)])        

        C = Z[self.batch_idx_all+(0,)]
        G = Z[self.batch_idx_all+(slice(1,1+n_dep_gens),)]
        Grest = Z[self.batch_idx_all+(slice(1+n_dep_gens,None),)]
        if expMat == None and id == None:
            # NOTE: MERGE redundant for 000?
            expMat = torch.eye(G.shape[self.batch_dim],dtype=torch.long) # if G is EMPTY_TENSOR, it will be EMPTY_TENSOR, size = (0,0)
            self.expMat,G = removeRedundantExponentsBatch(expMat,G,self.batch_idx_all,3)
            #self.expMat =expMat
            self.id = PROPERTY_ID.update(self.expMat.shape[1],prop) # if G is EMPTY_TENSOR, if will be EMPTY_TENSOR
        elif expMat != None:
            #check correctness of user input 
            if isinstance(expMat, list):
                expMat = torch.tensor(expMat)
            assert isinstance(expMat,torch.Tensor), 'The exponent matrix should be either torch tensor or list.'
            assert expMat.dtype in (torch.int, torch.long,torch.short), 'Exponent should have integer elements.'
            assert torch.all(expMat >= 0) and expMat.shape[0] == n_dep_gens, 'Invalid exponent matrix.' 
            self.expMat,G = removeRedundantExponentsBatch(expMat,G,self.batch_idx_all,3)
            #self.expMat =expMat
            if id != None:
                if isinstance(id, list):
                    id = torch.tensor(id,dtype=torch.long)
                if id.shape[0] !=0:
                    assert prop == 'None', 'Either ID or property should not be defined.'
                    assert max(id) < PROPERTY_ID.offset, 'Non existing ID is defined'
                assert isinstance(id, torch.Tensor), 'The identifier vector should be either torch tensor or list.'
                assert id.shape[0] == expMat.shape[1], f'Invalid vector of identifiers. The number of exponents is {expMat.shape[1]}, but the number of identifiers is {id.shape[0]}.'
                self.id = id
            else:
                self.id = PROPERTY_ID.update(self.expMat.shape[1],prop)
        elif isinstance(id, torch.Tensor) and id.shape[0] == 0:
            self.expMat = torch.eye(0,dtype=torch.long)
            self.id = id
        elif isinstance(id, list) and id.shape[0] == 0:
            self.expMat = torch.eye(0,dtype=torch.long)
            self.id = torch.tensor(id,dtype=torch.long)      
        else:
            assert False, 'Identifiers can only be defined as long as the exponent matrix is defined.'
        self.Z = torch.cat((C.unsqueeze(-3),G,Grest),dim=-3)
        self.n_dep_gens = G.shape[-3]
    def __getitem__(self,idx):
        Z = self.Z[idx]
        if len(Z.shape) > 3:
            return batchMatPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id)
        else:
            return zp.matPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id)
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
        return batchMatPolyZonotope(self.Z.transpose(-1,-2),self.n_dep_gens,self.expMat,self.id)
    def to(self,dtype=None,itype=None,device=None):
        Z = self.Z.to(dtype=dtype,device=device)
        expMat = self.expMat.to(dtype=itype,device=device)
        id = self.id.to(device=device)
        return batchMatPolyZonotope(Z,self.n_dep_gens,expMat,id)

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
            Z = self.Z @ other
            if len(other.shape) == 1:
                return batchPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id)
            else:
                return batchMatPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id)

        elif isinstance(other,(batchPolyZonotope,zp.polyZonotope)):
            assert self.n_cols == other.dimension
            id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)       
            _Z = (self.Z.unsqueeze(-3) @ other.Z.unsqueeze(-1).unsqueeze(-4)).squeeze(-1)
            Z_shape = _Z.shape[:-3]+(-1,self.n_rows)
            z1 = _Z[self.batch_idx_all +(slice(None,self.n_dep_gens+1),0)]
            z2 = _Z[self.batch_idx_all +(slice(None,self.n_dep_gens+1),slice(1,other.n_dep_gens+1))].reshape(Z_shape)
            z3 = _Z[self.batch_idx_all +(slice(self.n_dep_gens+1,None),)].reshape(Z_shape)
            z4 = _Z[self.batch_idx_all +(slice(None,self.n_dep_gens+1),slice(other.n_dep_gens+1,None))].reshape(Z_shape) 
            Z = torch.cat((z1,z2,z3,z4),dim=-2)
            expMat = torch.vstack((expMat1,expMat2,expMat2.repeat(self.n_dep_gens,1)+expMat1.repeat_interleave(other.n_dep_gens,dim=0)))
            n_dep_gens = (self.n_dep_gens+1) * (other.n_dep_gens+1)-1 
            return batchPolyZonotope(Z,n_dep_gens,expMat,id)

        elif isinstance(other,(batchMatPolyZonotope,matPolyZonotope)):
            assert self.n_cols == other.n_rows
            id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            _Z = self.Z.unsqueeze(-3) @ other.Z.unsqueeze(-4)
            Z_shape = _Z.shape[:-4]+(-1,self.n_rows,other.n_cols)            
            z1 = _Z[self.batch_idx_all +(slice(None,self.n_dep_gens+1),0)]
            z2 = _Z[self.batch_idx_all +(slice(None,self.n_dep_gens+1),slice(1,other.n_dep_gens+1))].reshape(Z_shape)
            z3 = _Z[self.batch_idx_all +(slice(self.n_dep_gens+1,None),)].reshape(Z_shape)
            z4 = _Z[self.batch_idx_all +(slice(None,self.n_dep_gens+1),slice(other.n_dep_gens+1,None))].reshape(Z_shape) 
            Z = torch.cat((z1,z2,z3,z4),dim=-3)
            expMat = torch.vstack((expMat1,expMat2,expMat2.repeat(self.n_dep_gens,1)+expMat1.repeat_interleave(other.n_dep_gens,dim=0)))
            n_dep_gens = (self.n_dep_gens+1) * (other.n_dep_gens+1)-1 
            return batchMatPolyZonotope(Z,n_dep_gens,expMat,id)
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
        if isinstance(other,torch.Tensor):
            assert other.shape[-1] == self.n_rows
            assert len(other.shape) > 2
            Z = other @ self.Z
            return batchMatPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id)
        elif isinstance(other,matPolyZonotope):
            assert other.n_cols == self.n_rows
            id, expMat1, expMat2 = mergeExpMatrix(other.id,self.id,other.expMat,self.expMat)
            _Z = other.Z.unsqueeze(-3) @ self.Z.unsqueeze(-4)
            Z_shape = _Z.shape[:-4]+(-1,other.n_rows,self.n_cols)

            z1 = _Z[self.batch_idx_all +(slice(None,other.n_dep_gens+1),0)]
            z2 = _Z[self.batch_idx_all +(slice(None,other.n_dep_gens+1),slice(1,self.n_dep_gens+1))].reshape(Z_shape)
            z3 = _Z[self.batch_idx_all +(slice(other.n_dep_gens+1,None),)].reshape(Z_shape)
            z4 = _Z[self.batch_idx_all +(slice(None,other.n_dep_gens+1),slice(self.n_dep_gens+1,None))].reshape(Z_shape) 
            Z = torch.cat((z1,z2,z3,z4),dim=-3)
            expMat = torch.vstack((expMat1,expMat2,expMat2.repeat(self.n_dep_gens,1)+expMat1.repeat_interleave(other.n_dep_gens,dim=0)))
            n_dep_gens = (self.n_dep_gens+1) * (other.n_dep_gens+1)-1 
            return batchMatPolyZonotope(Z,n_dep_gens,expMat,id)

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
            ind_red, ind_rem = ind[self.batch_idx_all+(slice(K),)], ind[self.batch_idx_all+(slice(K,None),)]
            # construct a zonotope from the gens that are removed
            Ztemp = zp.batchMatZonotope(torch.cat((torch.zeros(self.batch_shape+(1,)+self.shape),G.gather(-3,ind_rem)),dim=-3))
            # reduce the constructed zonotope with the reducetion techniques for linear zonotopes
            zonoRed = Ztemp.reduce(1,option)
            # add the reduced gens as new indep gens
            ZRed = torch.cat(((self.C + zonoRed.center).unsqueeze(-3),self.G,G.gather(-3,ind_red),zonoRed.generators),dim=-3)
        else:
            ZRed = self.Z
        n_dg_red = self.n_dep_gens
        if self.n_rows == 1 == self.n_cols and n_dg_red != 1:            
            ZRed = torch.cat((ZRed[self.batch_idx_all+(0,)],ZRed[self.batch_idx_all+(slice(1,n_dg_red+1),)].sum(-3).unsqueeze(-3),ZRed[self.batch_idx_all+(slice(n_dg_red+1,None),)]),dim=-3)
            n_dg_red = 1
        return batchMatPolyZonotope(ZRed,n_dg_red,self.expMat,self.id)