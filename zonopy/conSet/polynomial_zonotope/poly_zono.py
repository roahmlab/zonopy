"""
Define class for matrix polynomial zonotope
Author: Yongseok Kwon
Reference: CORA, Patrick Holme's implementation
"""
from zonopy.conSet.polynomial_zonotope.utils import removeRedundantExponents, mergeExpMatrix, pz_repr
from zonopy.conSet import DEFAULT_OPTS, PROPERTY_ID
import zonopy as zp
import torch

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
    dtype: data type of class properties
    , torch.float or torch.double
    itype: data type for exponent matrix
    , torch.int or torch.long or torch.short
    device: device for torch
    , 'cpu', 'gpu', 'cuda', ...

    Eq. (coeff. a1,a2,...,aN; b1,b2,...,bp \in [0,1])
    G = [gd1,gd2,...,gdN]
    Grest = [gi1,gi2,...,giM]
    expMat = [[i11,i12,...,i1N],[i21,i22,...,i2N],...,[ip1,ip2,...,ipN]]
    id = [0,1,2,...,p-1]

    pZ = c + a1*gi1 + a2*gi2 + ... + aN*giN + b1^i11*b2^i21*...*bp^ip1*gd1 + b1^i12*b2^i22*...*bp^ip2*gd2 + ... 
    + b1^i1M*b2^i2M*...*bp^ipM*gdM
     
    '''
    # NOTE: property for mat pz
    def __init__(self,c=EMPTY_TENSOR,G=EMPTY_TENSOR,Grest=EMPTY_TENSOR,expMat=None,id=None,dtype=None,itype=None,device=None,prop='None'):
        if dtype is None:
            dtype = DEFAULT_OPTS.DTYPE
        if itype is None:
            itype = DEFAULT_OPTS.ITYPE
        if device is None:
            device = DEFAULT_OPTS.DEVICE
        if type(c) == list:
            c = torch.tensor(c)
        if type(G) == list:
            G = torch.tensor(G)
        if type(Grest) == list:
            Grest = torch.tensor(Grest)
        if dtype == float:
            dtype = torch.double
        if itype == int:
            itype = torch.long
        assert isinstance(prop,str), 'Property should be string.'
        assert dtype == torch.float or dtype == torch.double, 'dtype should be float'
        assert itype == torch.int or itype == torch.long or itype == torch.short, 'itype should be integer'
        assert isinstance(c,torch.Tensor) and isinstance(G, torch.Tensor) and isinstance(Grest, torch.Tensor), 'The input matrix should be either torch tensor or list.'
        
        assert len(c.shape) == 1, f'The center should be a column tensor, but len(c.shape) is {len(c.shape)}'
        self.dimension = c.shape[0]
        assert G.numel() == 0 or self.dimension == G.shape[0], f'Dimension mismatch between center ({self.dimension}) and dependent generator matrix ({G.shape[0]}).'
        assert Grest.numel() == 0 or self.dimension == Grest.shape[0], f'Dimension mismatch between center ({self.dimension}) and dependent generator matrix ({Grest.shape[0]}).'

        self.c = c.to(dtype=dtype,device=device)
        self.G = G.reshape(self.dimension,-1 if G.numel() != 0 else 0).to(dtype=dtype,device=device)
        self.Grest = Grest.reshape(self.dimension,-1 if Grest.numel() != 0 else 0).to(dtype=dtype,device=device)

        if expMat == None and id == None:
            # NOTE: MERGE redundant for 000?
            expMat = torch.eye(self.G.shape[-1],dtype=itype,device=device) # if G is EMPTY_TENSOR, it will be EMPTY_TENSOR, size = (0,0)
            self.expMat,self.G = removeRedundantExponents(expMat,self.G)
            self.id = PROPERTY_ID.update(self.expMat.shape[0],prop,device) # if G is EMPTY_TENSOR, if will be EMPTY_TENSOR
            #self.id = torch.arange(self.expMat.shape[0],dtype=torch.long,device=device) 
        elif expMat != None:
            #check correctness of user input 
            if isinstance(expMat, list):
                expMat = torch.tensor(expMat)
            assert type(expMat) == torch.Tensor, 'The exponent matrix should be either torch tensor or list.'
            assert expMat.dtype == torch.int or expMat.dtype == torch.long or expMat.dtype == torch.short, 'Exponent should have integer elements.'
            assert torch.all(expMat >= 0) and expMat.shape[1] == self.G.shape[-1], 'Invalid exponent matrix.'
            expMat = expMat.to(dtype=itype,device=device)            
            self.expMat,self.G = removeRedundantExponents(expMat,self.G)
            #if self.G.numel()==0:
                #self.id = PROPERTY_ID.update(self.expMat.shape[0],prop,device)
                #self.id = torch.arange(self.expMat.shape[0],dtype=torch.long,device=device)
            if id != None:
                if isinstance(id, list):
                    id = torch.tensor(id)
                if id.numel() !=0:
                    assert prop == 'None', 'Either ID or property should not be defined.'
                    assert max(id) < PROPERTY_ID.offset, 'Non existing ID is defined'
                assert isinstance(id, torch.Tensor), 'The identifier vector should be either torch tensor or list.'
                assert id.numel() == expMat.shape[0], f'Invalid vector of identifiers. The number of exponents is {expMat.shape[0]}, but the number of identifiers is {id.numel()}.'
                self.id = id.to(dtype=torch.long,device=device)  
            else:
                self.id = PROPERTY_ID.update(self.expMat.shape[0],prop,device) 
        elif isinstance(id, torch.Tensor) and id.numel() == 0:
            self.expMat = torch.eye(0,dtype=itype,device=device)
            self.id = id.to(dtype=torch.long,device=device)  
        elif isinstance(id, list) and len(id) == 0:
            self.expMat = torch.eye(0,dtype=itype,device=device)
            self.id = torch.tensor(id,dtype=torch.long,device=device)      
        else:
            assert False, 'Identifiers can only be defined as long as the exponent matrix is defined.'
        self.__dtype, self.__itype, self.__device  = dtype, itype, device
    @property
    def dtype(self):
        return self.__dtype
    @property
    def itype(self):
        return self.__itype
    @property
    def device(self):
        return self.__device
    @property
    def n_generators(self):
        return self.G.shape[1] + self.Grest.shape[1]
    @property
    def n_dep_gens(self):
        return self.G.shape[1]
    @property
    def n_indep_gens(self):
        return self.Grest.shape[1]
    def to(self,dtype=None,itype=None,device=None):
        if dtype is None:
            dtype = self.dtype
        if itype is None:
            itype = self.itype
        if device is None:
            device = self.device
        return polyZonotope(self.c,self.G,self.Grest,self.expMat,self.id,dtype,itype,device)

    def __str__(self):
        if self.expMat.numel() == 0:
            expMat_print = EMPTY_TENSOR
        else:
            expMat_print = self.expMat[torch.argsort(self.id)]
        
        pz_str = f"""center: \n{self.c.to(dtype=torch.float,device='cpu')} \n\nnumber of dependent generators: {self.G.shape[-1]} 
            \ndependent generators: \n{self.G.to(dtype=torch.float,device='cpu')}  \n\nexponent matrix: \n {expMat_print.to(dtype=torch.long,device='cpu')}
            \nnumber of independent generators: {self.Grest.shape[-1]} \n\nindependent generators: \n {self.Grest.to(dtype=torch.float,device='cpu')}
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
        # TODO: allow to add bw polyZonotope and zonotope

        # if other is a vector
        if  isinstance(other,(torch.Tensor,float,int)):
            assert isinstance(other,(float,int)) or other.shape == self.c.shape or len(other.shape) == 0
            c = self.c + other
            G, Grest, expMat, id = self.G, self.Grest, self.expMat, self.id
        # if other is a polynomial zonotope
        elif isinstance(other,polyZonotope): # exact Plus
            id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            expMat, G = removeRedundantExponents(
                torch.hstack((expMat1, expMat2)),
                torch.hstack((self.G, other.G))
                )
            c = self.c + other.c
            Grest = torch.hstack((self.Grest,other.Grest))
            '''
            assert other.dimension == self.dimension
            c = self.c + other.c
            G = torch.hstack((self.G,other.G))
            Grest = torch.hstack((self.Grest,other.Grest))
            
            expMat = torch.block_diag(self.expMat,other.expMat)
            if self.id.numel() !=0:
                id_offset = max(self.id)+1
            else:
                id_offset = 0 
            id = torch.hstack((self.id,other.id+id_offset))
            
            [id, expMat1, expMat2] = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            '''
        return polyZonotope(c,G,Grest,expMat,id,self.__dtype,self.__itype,self.__device)
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
        return polyZonotope(-self.c, -self.G, self.Grest, self.expMat, self.id, self.__dtype, self.__itype, self.__device)

    def __iadd__(self,other): 
        return self+other
    def __isub__(self,other):
        return self-other
    def __mul__(self,other):
        if isinstance(other,(torch.Tensor,int,float)):
            #assert len(other.shape) == 1
            assert isinstance(other,(int,float)) or len(other.shape) == 0 or self.dimension == other.shape[0] or self.dimension == 1 or other.shape[0] == 1, 'Invalid dimension'
            c = self.c*other
            G = (self.G.T*other).T
            Grest = (self.Grest.T*other).T
            expMat = self.expMat
            id = self.id
            
        elif isinstance(other,polyZonotope):
            assert self.dimension == 1 and other.dimension == 1, 'Both polynomial zonotope must have dimension 1.'
            '''
            if self.id.numel() !=0:
                id_offset = max(self.id)+1
            else:
                id_offset = 0 
            id = torch.hstack((self.id,other.id+id_offset))
            expMat1 = torch.vstack((self.expMat,torch.zeros(other.expMat.shape[0],self.expMat.shape[1]))).to(dtype=self.__itype)
            expMat2 = torch.vstack((torch.zeros(self.expMat.shape[0],other.expMat.shape[1]),other.expMat)).to(dtype=self.__itype)
            '''
            [id, expMat1, expMat2] = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            G, Grest, expMat = EMPTY_TENSOR,EMPTY_TENSOR, EMPTY_TENSOR.reshape(id.numel(),0).to(dtype=self.__itype,device=self.__device)
            G1, G2, Grest1, Grest2 = self.G.reshape(-1), other.G.reshape(-1), self.Grest.reshape(-1), other.Grest.reshape(-1)            
            c = self.c*other.c 
            # deal with dependent generators
            if other.G.numel() != 0:
                c_g = self.c * other.G
                G = torch.hstack((G,c_g))
                expMat = torch.hstack((expMat,expMat2))
            if self.G.numel() != 0:
                g_c = self.G * other.c
                G = torch.hstack((G,g_c))
                expMat = torch.hstack((expMat,expMat1))
            if self.G.numel() != 0 and other.G.numel() != 0:
                g_g = torch.outer(G1,G2).reshape(1,-1)
                G = torch.hstack((G,g_g))
                expMat = torch.hstack((expMat, expMat1.repeat_interleave(expMat2.shape[1],dim=1)+expMat2.repeat(1,expMat1.shape[1])))
            # deal with independent generators
            if other.Grest.numel() != 0:
                c_grest = self.c*other.Grest
                Grest = torch.hstack((Grest,c_grest))
            if self.Grest.numel() != 0:
                grest_c = self.Grest*other.c
                Grest = torch.hstack((Grest,grest_c))
            if self.Grest.numel() != 0 and other.Grest.numel() != 0:
                grest_grest = torch.outer(Grest1,Grest2).reshape(1,-1)
                Grest = torch.hstack((Grest,grest_grest))
            if self.G.numel() !=0 and other.Grest.numel() !=0:
                g_grest = torch.outer(G1,Grest2).reshape(1,-1)
                Grest = torch.hstack((Grest,g_grest))
            if self.Grest.numel() != 0 and other.G.numel() !=0:
                grest_g = torch.outer(Grest1,G2).reshape(1,-1)
                Grest = torch.hstack((Grest,grest_g))
        
        return polyZonotope(c,G,Grest,expMat,id,self.__dtype,self.__itype,self.__device)
    __rmul__ = __mul__

    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a matrix or an interval matrix with a polyZonotope
        self: <polyZonotope>
        other: <torch.tensor> OR <intervals>
        return <polyZonotope>
        '''
        # TODO: Need to define intervals for matrix
        
        # if other is a matrix
        if isinstance(other, torch.Tensor):
            
            c = other@self.c
            G = other@self.G
            Grest = other@self.Grest
            

        # if other is an interval matrix

        return polyZonotope(c,G,Grest,self.expMat,self.id,self.__dtype,self.__itype,self.__device)
                 
    def reduce(self,order,option='girard'):
        # extract dimensions
        N = self.dimension
        P = self.G.shape[1]
        Q = self.Grest.shape[1]
            
        # number of gens kept (N gens will be added back after reudction)
        K = int(N*order-N)
        # check if the order need to be reduced
        if P+Q > N*order and K >=0:
            G = torch.hstack((self.G,self.Grest))
            # half the generators length for exponents that are all even
            temp = torch.prod(torch.ones(self.expMat.shape,device=self.device)-(self.expMat%2),0)
            ind = temp.nonzero().reshape(-1)
            G[:,ind] *= 0.5
            # caculate the length of the gens with a special metric
            len = torch.sum(G**2,0)
            # determine the smallest gens to remove            
            ind = torch.argsort(len,descending=True)
            ind_red,ind_rem = ind[:K], ind[K:]
            # split the indices into the ones for dependent and independent
            indDep = ind_rem[ind_rem < P].to(dtype=torch.long)
            indInd = ind_rem[ind_rem >= P].to(dtype=torch.long)
            indInd = indInd - P
            indDep_red = ind_red[ind_red < P].to(dtype=torch.long)
            indInd_red = ind_red[ind_red >= P].to(dtype=torch.long)
            indInd_red = indInd_red - P
            # construct a zonotope from the gens that are removed
            Grem = self.G[:,indDep]
            Erem = self.expMat[:,indDep]
            GrestRem = self.Grest[:,indInd]
            pZtemp = polyZonotope(torch.zeros(N),Grem,GrestRem,Erem,None,self.__dtype,self.__itype,self.__device)
            zono = pZtemp.to_zonotope() # zonotope over-approximation
            # reduce the constructed zonotope with the reducetion techniques for linear zonotopes
            zonoRed = zono.reduce(1,option)
            
            # remove the gens that got reduce from the gen matrices
            GRed = self.G[:,indDep_red]
            expMatRed = self.expMat[:,indDep_red]
            GrestRed = self.Grest[:,indInd_red]
            
            # add the reduced gens as new indep gens
            cRed = self.c + zonoRed.center
            GrestRed = torch.hstack((GrestRed,zonoRed.generators))
        else:
            cRed = self.c
            GRed= self.G
            GrestRed = self.Grest
            expMatRed = self.expMat
        # remove all exponent vector dimensions that have no entries
        ind = torch.sum(expMatRed,1)>0
        #ind = temp.nonzero().reshape(-1)
        expMatRed = expMatRed[ind,:].to(dtype=torch.long)
        idRed = self.id[ind]

        if self.dimension == 1:
            GrestRed = torch.sum(GrestRed,dim=-1).reshape(1,-1)
        return polyZonotope(cRed,GRed,GrestRed,expMatRed,idRed,self.__dtype,self.__itype,self.__device)

    def exactCartProd(self,other):
        '''
        self: <polyZonotope>
        other: <polyZonotope>
        return <polyZonotope>
        '''    
        if isinstance(other,polyZonotope):
            dim1, dim2 = self.dimension, other.dimension 
            c = torch.hstack((self.c,other.c))
            if self.G.numel() == 0:
                if other.G.numel() == 0:
                    G = EMPTY_TENSOR
                    expMat = None
                    id = None
                else:
                    G = torch.vstack((torch.zeros(dim1,other.n_dep_gens),other.G))
                    expMat = other.expMat
                    id = other.id
            else:
                if other.G.numel() == 0:
                    G = torch.vstack((self.G,torch.zeros(dim2,self.n_dep_gens)))
                    expMat = self.expMat
                    id = self.id
                else:
                    id,expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
                    G = torch.block_diag(self.G,other.G)
                    expMat = torch.hstack((expMat1,expMat2))
            
            if self.Grest.numel() == 0:
                if other.Grest.numel() == 0:
                    Grest = EMPTY_TENSOR
                else:
                    Grest = torch.hstack((torch.zeros(dim1,other.n_indep_gens),other.Grest))
            else:
                if other.Grest.numel() == 0:
                    Grest = torch.hstack((self.Grest, torch.zeros(dim2,self.n_indep_gens)))
                else:
                    Grest = torch.block_diag(self.Grest,other.Grest)
        return polyZonotope(c,G,Grest,expMat,id,self.__dtype,self.__itype, self.__device)

    def to_zonotope(self):
        if self.G.numel() != 0:
            temp = torch.prod(torch.ones(self.expMat.shape)-(self.expMat%2),0)
            Gquad = self.G[:,temp==1]

            c = self.c + 0.5*torch.sum(Gquad,1)
            G = torch.hstack((self.G[:,temp==0],0.5*Gquad,self.Grest))
            Z = torch.hstack((c.reshape(-1,1),G))
        else:
            Z = torch.hstack((self.c.reshape(-1,1),self.Grest))
        return zp.zonotope(Z,self.__dtype,self.__device)

    def to_interval(self,method='interval'):
        if method == 'interval':
            return self.to_zonotope().to_interval()
        else:
            assert False, 'Not implemented'

    @property
    def Z(self):
        return torch.hstack((self.c.reshape(-1,1),self.G,self.Grest)) 

    def slice_dep(self,id_slc,val_slc):
        '''
        Slice polynomial zonotpe in depdent generators
        id_slc: id to dlice
        val_slc: indeterminant to slice
        '''
        if isinstance(id_slc,(int,list)):
            if isinstance(id_slc,int):
                id_slc = [id_slc]
            id_slc = torch.Tensor(id_slc)
        if isinstance(val_slc,(int,float,list)):
            if isinstance(val_slc,(int,float)):
                val_slc = [val_slc]
            val_slc = torch.Tensor(val_slc)
        
        if any(abs(val_slc)>1):
            import pdb; pdb.set_trace()
        #assert all(val_slc<=1) and all(val_slc>=-1), 'Indereminant should be in [-1,1].'
        
        id_slc, val_slc = id_slc.reshape(-1,1), val_slc.reshape(-1,1)
        order = torch.argsort(id_slc.reshape(-1))
        id_slc, val_slc  = id_slc[order], val_slc[order]
        ind = torch.any(self.id==id_slc,dim=0)#.nonzero().reshape(-1)
        ind2 = torch.any(self.id==id_slc,dim=1)#.nonzero().reshape(-1)        
        #assert ind.numel()==len(id_slc), 'Some specidied IDs do not exist!'
        if ind.numel() != 0:
            G = self.G*torch.prod(val_slc[ind2]**self.expMat[ind],dim=0)
            expMat = torch.clone(self.expMat)
            expMat[ind] = 0
        else:
            expMat = torch.clone(self.expMat)
            G = torch.clone(self.G)

        expMat, G = removeRedundantExponents(expMat,G)
        ind = torch.sum(expMat,0) == 0
        if torch.any(ind):
            c = self.c + torch.sum(G[:,ind],1)
            G = G[:,~ind]
            expMat = expMat[:,~ind]
        else:
            c = self.c
        
        id = self.id
        ind = torch.sum(expMat,1) == 0
        if torch.any(ind):
            expMat = expMat[~ind]
            id = id[~ind]
        if G.numel() == 0 and self.Grest.numel() == 0:
            return polyZonotope(c,dtype=self.__dtype,itype=self.__itype,device=self.__device)
        else:
            return polyZonotope(c,G,self.Grest,expMat,id,self.__dtype,self.__itype,self.__device)
    
    def deleteZerosGenerators(self,eps=0):
        expMat, G = removeRedundantExponents(self.expMat,self.G,eps)
        ind = torch.sum(expMat,0) == 0
        if torch.any(ind):
            c = self.c + torch.sum(G[:,ind],1)
            G = G[:,~ind]
            expMat = expMat[:,~ind]
        else:
            c = self.c
        
        id = self.id
        ind = torch.sum(expMat,1) == 0
        if torch.any(ind):
            expMat = expMat[~ind]
            id = id[~ind]
        if G.numel() == 0 and self.Grest.numel() == 0:
            return polyZonotope(c,dtype=self.__dtype,itype=self.__itype,device=self.__device)
        else:
            return polyZonotope(c,G,self.Grest,expMat,id,self.__dtype,self.__itype,self.__device)        
        
    def project(self,dim=[0,1]):
        c = self.c[dim,:]
        G = self.G[dim,:]
        Grest = self.Grest[dim,:]
        return polyZonotope(c,G,Grest,self.expMat,self.id,self.__dtype,self.__itype,self.__device)
    def plot(self,dim=[0,1]):
        # NOTE: delte zero generators
        pz = self.project(dim)
        

if __name__ == '__main__':
    #pz = polyZonotope(torch.tensor([1.212,24142.42]),torch.eye(2),torch.eye(2),dtype=float,itype=int)
    #print(pz.__repr__())
    n=10
    pz = polyZonotope(torch.tensor([1]),torch.arange(n).reshape(1,n))
    print(pz)
    
    print(pz.reduce(10))







        