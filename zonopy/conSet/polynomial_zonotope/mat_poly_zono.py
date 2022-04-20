"""
Define class for matrix polynomial zonotope
Author: Yongseok Kwon
Reference: Patrick Holme's implementation
"""
from zonopy.conSet.polynomial_zonotope.utils import removeRedundantExponents, mergeExpMatrix
from zonopy.conSet import DEFAULT_OPTS, PROPERTY_ID
from zonopy import polyZonotope
from zonopy.conSet.utils import G_mul_c, G_mul_g, C_mul_G, G_mul_C, G_mul_G
import zonopy as zp
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
    def __init__(self,C,G=EMPTY_TENSOR,Grest=EMPTY_TENSOR,expMat=None,id=None,dtype=None,itype=None,device=None,prop='None'):
        if dtype is None:
            dtype = DEFAULT_OPTS.DTYPE
        if itype is None:
            itype = DEFAULT_OPTS.ITYPE
        if device is None:
            device = DEFAULT_OPTS.DEVICE
        
        if type(C) == list:
            C = torch.tensor(C)
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
        assert isinstance(C,torch.Tensor) and isinstance(G, torch.Tensor) and isinstance(Grest, torch.Tensor), 'The input matrix should be either torch tensor or list.'

        assert len(C.shape) == 2, f'The center should be a matrix tensor, but len(c.shape) is {len(C.shape)}.'
        self.n_rows, self.n_cols = C.shape
        assert G.numel() == 0 or (self.n_rows == G.shape[0] and self.n_cols == G.shape[1]),f'Matrix dimension mismatch between center ([{self.n_rows}, {self.n_cols}]) and dependent generator matrix ([{G.shape[0]}, {G.shape[1]}]).'
        assert Grest.numel() == 0 or (self.n_rows == Grest.shape[0] and self.n_cols == Grest.shape[1]), f'Matrix dimension mismatch between center ([{self.n_rows}, {self.n_cols}]) and independent generator matrix ([{Grest.shape[0]}, {Grest.shape[1]}]).'

        self.C = C.to(dtype=dtype,device=device)
        self.G = G.reshape(self.n_rows,self.n_cols,-1 if G.numel() != 0 else 0).to(dtype=dtype,device=device)
        self.Grest = Grest.reshape(self.n_rows,self.n_cols,-1 if Grest.numel() != 0 else 0).to(dtype=dtype,device=device)

        if expMat == None and id == None:
            # NOTE: MERGE redundant for 000?
            expMat = torch.eye(self.G.shape[-1],dtype=itype,device=device) # if G is EMPTY_TENSOR, it will be EMPTY_TENSOR, size = (0,0)
            self.expMat,self.G = removeRedundantExponents(expMat,self.G)
            self.id = PROPERTY_ID.update(self.expMat.shape[0],prop,device) # if G is EMPTY_TENSOR, if will be EMPTY_TENSOR
        elif expMat != None:
            #check correctness of user input
            if isinstance(expMat, list):
                expMat = torch.tensor(expMat)
            assert type(expMat) == torch.Tensor, 'The exponent matrix should be either torch tensor or list.'
            assert expMat.dtype == torch.int or expMat.dtype == torch.long or expMat.dtype == torch.short, 'Exponent should have integer elements.'
            assert torch.all(expMat >= 0) and expMat.shape[1] == self.G.shape[-1], 'Invalid exponent matrix.'
            expMat = expMat.to(dtype=itype,device=device)            
            self.expMat,self.G = removeRedundantExponents(expMat,self.G)
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
            raise ValueError('Identifiers can only be defined as long as the exponent matrix is defined.')
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
        return self.G.shape[-1] + self.Grest.shape[-1]
    @property
    def n_dep_gens(self):
        return self.G.shape[-1]
    @property
    def n_indep_gens(self):
        return self.Grest.shape[-1]
    @property
    def T(self):
        return matPolyZonotope(self.C.T,self.G.permute(1,0,2),self.Grest.permute(1,0,2),self.expMat,self.id,self.__dtype,self.__itype,self.__device)

    def to(self,dtype=None,itype=None,device=None):
        if dtype is None:
            dtype = self.dtype
        if itype is None:
            itype = self.itype
        if device is None:
            device = self.device
        return matPolyZonotope(self.C,self.G,self.Grest,self.expMat,self.id,dtype,itype,device)

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
            if len(other.shape) == 1:
                assert other.shape[0] == self.n_cols
                c = self.C @ other
                G = G_mul_c(self.G,other)
                Grest = G_mul_c(self.Grest,other)    
                id = self.id
                expMat = self.expMat   
                return polyZonotope(c,G,Grest,expMat,id,self.__dtype,self.__itype,self.__device)
            elif len(other.shape) == 2:
                assert other.shape[0] == self.n_cols
                C = self.C @ other
                G = (self.G.permute(2,0,1) @ other).permute(1,2,0)
                Grest = (self.Grest.permute(2,0,1) @ other).permute(1,2,0)
                id = self.id
                expMat = self.expMat   
                return matPolyZonotope(C,G,Grest,expMat,id,self.__dtype,self.__itype,self.__device)
            else:
                assert False, 'The other object should be 1 or 2-D tensor.'  
            
   

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
            id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            
            empty_tensor = EMPTY_TENSOR.to(device=self.__device)
            G, Grest, expMat = empty_tensor, empty_tensor, empty_tensor.reshape(id.numel(),0).to(dtype=self.__itype)
            
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
                expMat = torch.hstack((expMat, expMat1.repeat_interleave(expMat2.shape[1],dim=1)+expMat2.repeat(1,expMat1.shape[1])))
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
            return polyZonotope(c,G,Grest,expMat.to(dtype=torch.long),id,self.__dtype,self.__itype,self.__device)

        elif type(other) == matPolyZonotope:
            assert self.n_cols == other.n_rows
            id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            dims = [self.n_rows, self.n_cols, other.n_cols]
            empty_tensor = EMPTY_TENSOR.to(device=self.__device)
            G, Grest, expMat = empty_tensor,empty_tensor, empty_tensor.reshape(id.numel(),0).to(dtype=self.__itype)
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
                expMat = torch.hstack((expMat, expMat1.repeat_interleave(expMat2.shape[1],dim=1)+expMat2.repeat(1,expMat1.shape[1])))
            
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
            return matPolyZonotope(C,G,Grest,expMat,id,self.__dtype,self.__itype,self.__device)


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
            return matPolyZonotope(C,G,Grest,self.expMat,self.id,self.__dtype,self.__itype,self.__device)

    def to_matZonotope(self):
        if self.G.numel() != 0:
            temp = torch.prod(torch.ones(self.expMat.shape,device=self.__device)-(self.expMat%2),0)
            Gquad = self.G[:,:,temp==1]
            c = self.C + 0.5*torch.sum(Gquad,-1)
            G = torch.cat((self.G[:,:,temp==0],0.5*Gquad,self.Grest),-1)
            Z = torch.cat((c.reshape(self.n_rows,self.n_cols,1),G),-1)
        else:
            Z = torch.cat((self.C.reshape(self.n_rows,self.n_cols,1),self.Grest),-1)
        #Z = torch.cat((self.C.reshape(self.n_rows,self.n_cols,1),self.G),-1)
        #Z = torch.cat((Z,self.Grest),-1)
        return zp.matZonotope(Z,self.__dtype,self.__device)

    def reduce(self,order,option='girard'):
        # extract dimensions
        N = self.n_rows * self.n_cols
        P = self.n_dep_gens
        Q = self.n_indep_gens
            
        # number of gens kept (N gens will be added back after reudction)
        K = int(N*order-N)
        # check if the order need to be reduced
        if P+Q > N*order and K >=0:
            G = torch.cat((self.G,self.Grest),-1)
            # half the generators length for exponents that are all even
            temp = torch.prod(torch.ones(self.expMat.shape,device=self.__device)-(self.expMat%2),0)
            ind = temp.nonzero().reshape(-1)
            G[:,:,ind] *= 0.5
            # caculate the length of the gens with a special metric
            len = torch.sum(G**2,(0,1))
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
            Grem = self.G[:,:,indDep]
            Erem = self.expMat[:,indDep]
            GrestRem = self.Grest[:,:,indInd]

            pZtemp = matPolyZonotope(torch.zeros(self.n_rows,self.n_cols),Grem,GrestRem,Erem,None,self.__dtype,self.__itype,self.__device)
            zono = pZtemp.to_matZonotope() # zonotope over-approximation
            # reduce the constructed zonotope with the reducetion techniques for linear zonotopes
            zonoRed = zono.reduce(1,option)
            
            # remove the gens that got reduce from the gen matrices
            GRed = self.G[:,:,indDep_red]
            expMatRed = self.expMat[:,indDep_red]
            GrestRed = self.Grest[:,:,indInd_red]
            
            # add the reduced gens as new indep gens
            cRed = self.C + zonoRed.center
            GrestRed = torch.cat((GrestRed,zonoRed.generators),-1)
            
        else:
            cRed = self.C
            GRed= self.G
            GrestRed = self.Grest
            expMatRed = self.expMat
        # remove all exponent vector dimensions that have no entries
        ind = torch.sum(expMatRed,1)>0
        #ind = temp.nonzero().reshape(-1)
        expMatRed = expMatRed[ind,:].to(dtype=int)
        idRed = self.id[ind]

        if self.n_rows == 1 and self.n_cols == 1:
            GrestRed = torch.sum(GrestRed,dim=-1).reshape(1,1,-1)
        return matPolyZonotope(cRed,GRed,GrestRed,expMatRed,idRed,self.__dtype,self.__itype,self.__device)

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
