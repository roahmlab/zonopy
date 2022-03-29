"""
Define class for polynomial zonotope
Reference: CORA
Writer: Yongseok Kwon
"""
from tempfile import tempdir
from zonopy.conSet.polynomial_zonotope.utils import removeRedundantExponents, mergeExpMatrix, pz_repr
from zonopy.conSet import DEFAULT_DTYPE, DEFAULT_ITYPE, DEFAULT_DEVICE
from zonopy.conSet.utils import delete_column
import zonopy as zp
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
    def __init__(self,c=EMPTY_TENSOR,G=EMPTY_TENSOR,Grest=EMPTY_TENSOR,expMat=None,id=None,dtype=DEFAULT_DTYPE,itype=DEFAULT_ITYPE,device=DEFAULT_DEVICE):
        # TODO: assign device
        # TODO: assign dtype for ind, exp
        # NOTE: ind might be better to be list

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

        assert dtype == torch.float or dtype == torch.double, 'dtype should be float'
        assert itype == torch.int or itype == torch.long or itype == torch.short, 'itype should be integer'
            
        assert type(c) == torch.Tensor and type(G) == torch.Tensor and type(Grest) == torch.Tensor, 'The input matrix should be either torch tensor or list.'
        
        self.dimension = c.shape[0]
        
        if len(c.shape) != 1:
            raise ValueError(f'The center should be a column tensor, but len(c.shape) is {len(c.shape)}')
        if G.numel() != 0 and self.dimension != G.shape[0]:
            raise ValueError(f'Dimension mismatch between center ({self.dimension}) and dependent generator matrix ({G.shape[0]}).')
        if Grest.numel() != 0 and self.dimension != Grest.shape[0]:
            raise ValueError(f'Dimension mismatch between center ({self.dimension}) and dependent generator matrix ({Grest.shape[0]}).')
        if expMat is not None and expMat.numel() == 0:
            expMat = None
        
        self.c = c.to(dtype=dtype,device=device)
        self.G = G.reshape(self.dimension,-1 if G.numel() != 0 else 0).to(dtype=dtype,device=device)
        self.Grest = Grest.reshape(self.dimension,-1 if Grest.numel() != 0 else 0).to(dtype=dtype,device=device)

        if expMat == None and id == None:

            # NOTE: MERGE redundant for 000?
            expMat = torch.eye(self.G.shape[-1],dtype=itype,device=device) # if G is EMPTY_TENSOR, it will be EMPTY_TENSOR, size = (0,0)
            self.expMat,self.G = removeRedundantExponents(expMat,self.G)
            self.id = torch.arange(self.G.shape[-1],dtype=int,device=device) # if G is EMPTY_TENSOR, if will be EMPTY_TENSOR
        elif expMat != None:
            #check correctness of user input
            if type(expMat) == list:
                expMat = torch.tensor(expMat)
            assert type(expMat) == torch.Tensor, 'The exponent matrix should be either torch tensor or list.'
            assert expMat.dtype == torch.int or expMat.dtype == torch.long or expMat.dtype == torch.short, 'Exponent should have integer elements.'
            assert torch.all(expMat >= 0) and expMat.shape[1] == self.G.shape[-1], 'Invalid exponent matrix.'
            expMat = expMat.to(dtype=itype,device=device)            
            expMat,G = removeRedundantExponents(expMat,self.G)
            self.G = G
            self.expMat =expMat
            if self.G.numel()==0:
                self.id = torch.arange(expMat.shape[0],dtype=int,device=device)
            elif id != None:
                if type(id) == list:
                    id = torch.tensor(id)
                assert type(id) == torch.Tensor, 'The identifier vector should be either torch tensor or list.'
                assert id.shape[0] == expMat.shape[0], f'Invalid vector of identifiers. The number of exponents is {expMat.shape[0]}, but the number of identifiers is {id.shape[0]}.'
                self.id = id.to(dtype=int,device=device)  
        elif type(id) == torch.Tensor and id.numel() == 0:
            self.expMat = torch.eye(0,dtype=itype,device=device)
            self.id = id.to(dtype=int,device=device)  
        elif type(id) == list and len(id) == 0:
            self.expMat = torch.eye(0,dtype=itype,device=device)
            self.id = torch.tensor(id,dtype=int,device=device)      
        else:
            raise ValueError('Identifiers can only be defined as long as the exponent matrix is defined.')
        self.dtype, self.itype, self.device  = dtype, itype, device

    @property
    def n_generators(self):
        return self.G.shape[1] + self.Grest.shape[1]
    @property
    def n_dep_gens(self):
        return self.G.shape[1]
    @property
    def n_indep_gens(self):
        return self.Grest.shape[1]


    def __str__(self):
        if self.expMat.numel() == 0:
            expMat_print = EMPTY_TENSOR
        else:
            expMat_print = self.expMat[torch.argsort(self.id)]
        
        pz_str = f"""center: \n{self.c.to(dtype=torch.float,device='cpu')} \n\nnumber of dependent generators: {self.G.shape[-1]} 
            \ndependent generators: \n{self.G.to(dtype=torch.float,device='cpu')}  \n\nexponent matrix: \n {expMat_print.to(dtype=int,device='cpu')}
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
        other: <np.ndarray> or <torch.tensor> OR <zonotope> OR <polyZonotope>
        return <polyZonotope>
        '''
        # TODO: allow to add bw polyZonotope and zonotope

        # if other is a vector
        if  type(other) == torch.Tensor:
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
            if self.id.numel() !=0:
                id_offset = max(self.id)
            else:
                id_offset = 0 
            id = torch.hstack((self.id,other.id+id_offset+1))
        return polyZonotope(c,G,Grest,expMat,id)
    
    __radd__ = __add__
    def __mul__(self,other):
        if isinstance(other,torch.Tensor):
            #assert len(other.shape) == 1
            assert self.dimension == other.shape[0] or self.dimension == 1 or other.shape[0] == 1, 'Invalid dimension'
            c = self.c*other
            G = (self.G.T*other).T
            Grest = (self.Grest.T*other).T
            
        elif isinstance(other,polyZonotope):
            assert False    
        return polyZonotope(c,G,Grest,self.expMat,self.id,self.dtype,self.itype,self.device)
    __rmul__ = __mul__

    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a matrix or an interval matrix with a polyZonotope
        self: <polyZonotope>
        other: <np.ndarray> or <torch.tensor> OR <intervals>
        return <polyZonotope>
        '''
        # TODO: Need to define intervals for matrix
        
        # if other is a matrix
        if type(other) == np.ndarray or type(other) == torch.Tensor:
            if type(other) == np.ndarray:
                other = torch.from_numpy(other)
            
            c = other@self.c
            G = other@self.G
            Grest = other@self.Grest
            
    
        # if other is an interval matrix

        return polyZonotope(c,G,Grest,self.expMat,self.id)
                 
    def reduce(self,order,option='girard'):
        # extract dimensions
        N = self.dimension
        P = self.G.shape[1]
        Q = self.Grest.shape[1]
            
        # number of gens kept (N gens will be added back after reudction)
        K = max(0,int(N*order-N))
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
            ind = ind[K:]
            # split the indices into the ones for dependent and independent
            indDep = ind[ind < P].to(dtype=int)
            indInd = ind[ind >= P].to(dtype=int)
            indInd = indInd - P
            # construct a zonotope from the gens that are removed
            Grem = self.G[:,indDep]
            Erem = self.expMat[:,indDep]
            GrestRem = self.Grest[:,indInd]
            pZtemp = polyZonotope(torch.zeros(N),Grem,GrestRem,Erem,None,self.dtype,self.itype,self.device)
            zono = pZtemp.to_zonotope() # zonotope over-approximation
            # reduce the constructed zonotope with the reducetion techniques for linear zonotopes
            zonoRed = zono.reduce(1,option)
            
            # remove the gens that got reduce from the gen matrices
            GRed = delete_column(self.G,indDep)
            expMatRed = delete_column(self.expMat,indDep)
            GrestRed = delete_column(self.Grest,indInd)
            
            # add the reduced gens as new indep gens
            cRed = self.c + zonoRed.center
            GrestRed = torch.hstack((GrestRed,zonoRed.generators))
            
        else:
            cRed = self.c
            GRed= self.G
            GrestRed = self.Grest
            expMatRed = self.expMat
        # remove all exponent vector dimensions that have no entries
        temp = torch.sum(expMatRed,1)>0
        ind = temp.nonzero().reshape(-1)
        expMatRed = expMatRed[ind,:].to(dtype=int)
        idRed = self.id[ind]

        return polyZonotope(cRed,GRed,GrestRed,expMatRed,idRed,self.dtype,self.itype,self.device)
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

    def to_zonotope(self):
        if self.G.numel() != 0:
            temp = torch.prod(torch.ones(self.expMat.shape)-(self.expMat%2),0)
            Gquad = self.G[:,temp==1]

            c = self.c + 0.5*torch.sum(Gquad,1)
            G = torch.hstack((self.G[:,temp==0],0.5*Gquad,self.Grest))
            Z = torch.hstack((c.reshape(-1,1),G))
        else:
            Z = torch.hstack((self.c.reshape(-1,1),self.Grest))
        return zp.zonotope(Z)
    @property
    def Z(self):
        return torch.hstack((self.c.reshape(-1,1),self.G,self.Grest)) 

if __name__ == '__main__':
    #pz = polyZonotope(torch.tensor([1.212,24142.42]),torch.eye(2),torch.eye(2),dtype=float,itype=int)
    #print(pz.__repr__())
    n=10
    pz = polyZonotope(torch.tensor([1]),torch.arange(n).reshape(1,n))
    print(pz)
    
    print(pz.reduce(10))







        