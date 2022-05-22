"""
Define class for zonotope
Author: Yongseok Kwon
Reference: CORA
"""
import torch
import matplotlib.patches as patches
from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope
from zonopy.conSet.interval.interval import interval
from zonopy.conSet.zonotope.utils import pickedGenerators, ndimCross

class zonotope:
    '''
    zono: <zonotope>, <torch.float64>

    Z: <torch.Tensor> center vector and generator matrix Z = [c,G]
    , shape [N+1, nx]
    center: <torch.Tensor> center vector
    , shape [nx] 
    generators: <torch.Tensor> generator matrix
    , shape [N, nx]
    dtype: data type of class properties
    , torch.float or torch.double
    device: device for torch
    , 'cpu', 'gpu', 'cuda', ...
    
    Eq. (coeff. a1,a2,...,aN \in [0,1])
    G = [[g1],[g2],...,[gN]]
    zono = c + a1*g1 + a2*g2 + ... + aN*gN
    '''
    def __init__(self,Z):
        ################ may not need these for speed ################ 
        if isinstance(Z,list):
            Z = torch.tensor(Z,dtype=torch.float)
        assert isinstance(Z,torch.Tensor), f'The input matrix should be either torch tensor or list, not {type(Z)}.'
        assert Z.dtype == torch.float or Z.dtype == torch.double, f'dtype should be either torch.float (torch.float32) or torch.double (torch.float64), but {Z.dtype}.'
        assert len(Z.shape) == 2, f'The dimension of Z input should be either 1 or 2, not {len(Z.shape)}.'
        ############################################################## 
        self.Z = Z
    @property
    def dtype(self):
        return self.Z.dtype
    @property
    def device(self):
        return self.Z.device
    @property
    def center(self):
        return self.Z[0]
    @center.setter
    def center(self,value):
        self.Z[0] = value
    @property
    def generators(self):
        return self.Z[1:]
    @generators.setter
    def generators(self,value):
        self.Z[1:] = value
    @property 
    def shape(self):
        return (self.Z.shape[1],)
    @property
    def dimension(self):
        return self.Z.shape[1]
    @property
    def n_generators(self):
        return len(self.Z)-1
    def to(self,dtype=None,device=None):    
        Z = self.Z.to(dtype=dtype, device=device)
        return zonotope(Z)
    
    def __str__(self):
        zono_str = f"""center: \n{self.center} \n\nnumber of generators: {self.n_generators} 
            \ngenerators: \n{self.generators} \n\ndimension: {self.dimension}\ndtype: {self.dtype} \ndevice: {self.device}"""
        del_dict = {'tensor':' ','    ':' ','(':'',')':''}
        for del_el in del_dict.keys():
            zono_str = zono_str.replace(del_el,del_dict[del_el])
        return zono_str
    def __repr__(self):
        return str(self.Z).replace('tensor','zonotope')
    def  __add__(self,other):
        '''
        Overloaded '+' operator for Minkowski sum
        self: <zonotope>
        other: <torch.tensor> OR <zonotope>
        return <polyZonotope>
        '''   
        if isinstance(other, torch.Tensor):
            Z = torch.clone(self.Z)
            assert other.shape == self.shape, f'array dimension does not match: should be {self.shape}, not {other.shape}.'
            Z[0] += other
        elif isinstance(other, zonotope): 
            assert self.dimension == other.dimension, f'zonotope dimension does not match: {self.dimension} and {other.dimension}.'
            Z = torch.vstack((self.center + other.center,self.generators,other.generators))
        else:
            assert False, f'the other object is neither a zonotope nor a torch tensor, not {type(other)}.'
        return zonotope(Z)

    __radd__ = __add__
    def __sub__(self,other):
        if isinstance(other, torch.Tensor):
            Z = torch.clone(self.Z)
            assert other.shape == self.shape, f'array dimension does not match: should be {self.shape}, not {other.shape}.'
            Z[0] -= other
        elif isinstance(other, zonotope): 
            assert self.dimension == other.dimension, f'zonotope dimension does not match: {self.dimension} and {other.dimension}.'
            Z = torch.vstack((self.center - other.center,self.generators,other.generators))
        else:
            assert False, f'the other object is neither a zonotope nor a torch tensor, not {type(other)}.'
        return zonotope(Z)
    def __rsub__(self,other):
        return -self.__sub__(other)
    def __iadd__(self,other):
        return self+other
    def __isub__(self,other):
        return self-other
    def __pos__(self):
        return self    
    
    def __neg__(self):
        '''
        Overloaded unary '-' operator for negation
        self: <zonotope>
        return <zonotope>
        '''   
        Z = -self.Z
        Z[1:] = self.Z[1:]
        return zonotope(Z)    
    
    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for matrix multiplication
        self: <zonotope>
        other: <torch.tensor>
        
        zono = other @ self

        return <zonotope>
        '''   
        assert isinstance(other, torch.Tensor), f'The other object should be torch tensor, but {type(other)}.'
        Z = self.Z@other.T
        return zonotope(Z)

    def slice(self,slice_dim,slice_pt):
        '''
        slice zonotope on specified point in a certain dimension
        self: <zonotope>
        slice_dim: <torch.Tensor> or <list> or <int>
        , shape  []
        slice_pt: <torch.Tensor> or <list> or <float> or <int>
        , shape  []
        return <zonotope>
        '''
        if isinstance(slice_dim, list):
            slice_dim = torch.tensor(slice_dim,dtype=torch.long)
        elif isinstance(slice_dim, int) or (isinstance(slice_dim, torch.Tensor) and len(slice_dim.shape)==0):
            slice_dim = torch.tensor([slice_dim],dtype=torch.long)

        if isinstance(slice_pt, list):
            slice_pt = torch.tensor(slice_pt,dtype=self.dtype)
        elif isinstance(slice_pt, int) or isinstance(slice_pt, float) or (isinstance(slice_pt, torch.Tensor) and len(slice_pt.shape)==0):
            slice_pt = torch.tensor([slice_pt],dtype=self.dtype)

        assert isinstance(slice_dim, torch.Tensor) and isinstance(slice_pt, torch.Tensor), 'Invalid type of input'
        assert len(slice_dim.shape) ==1, 'slicing dimension should be 1-dim component.'
        assert len(slice_pt.shape) ==1, 'slicing point should be 1-dim component.'
        assert len(slice_dim) == len(slice_pt), f'The number of slicing dimension ({len(slice_dim)}) and the number of slicing point ({len(slice_dim)}) should be the same.'

        slice_dim, ind = torch.sort(slice_dim)
        slice_pt = slice_pt[ind]

        c = self.center
        G = self.generators

        non_zero_idx = G[:,slice_dim] != 0
        assert torch.all(torch.sum(non_zero_idx,0)==1), 'There should be one generator for each slice index.'
        slice_idx = non_zero_idx.T.nonzero()[:,1]

        slice_c = c[slice_dim]
        slice_g = G[slice_idx,slice_dim]
        slice_lambda = (slice_pt-slice_c)/slice_g
        assert not (abs(slice_lambda)>1).any(), 'slice point is ouside bounds of reach set, and therefore is not verified'
        
        Z = torch.vstack((c + slice_lambda@G[slice_idx],G[~non_zero_idx.any(-1)]))
        return zonotope(Z)

    def project(self,dim=[0,1]):
        '''
        the projection of a zonotope onto the specified dimensions
        self: <zonotope>
        dim: <int> or <list> or <torch.Tensor> dimensions for prjection 
        
        return <zonotope>
        '''
        Z = self.Z[:,dim]
        return zonotope(Z)

    def polygon(self):
        '''
        converts a 2-d zonotope into a polygon as vertices
        self: <zonotope>

        return <torch.Tensor>, <torch.float64>
        '''
        dim = 2
        z = self.deleteZerosGenerators()
        c = z.center
        G = torch.clone(z.generators)
        n = z.n_generators
        x_max = torch.sum(abs(G[:,0]))
        y_max = torch.sum(abs(G[:,1]))
        
        G[z.generators[:,1]<0,:] = - z.generators[z.generators[:,1]<0,:] # make all y components as positive
        angles = torch.atan2(G[:,1], G[:,0])
        ang_idx = torch.argsort(angles)
                
        vertices_half = torch.vstack((torch.zeros(dim),2*G[ang_idx].cumsum(axis=0)))
        vertices_half[:,0] += x_max - torch.max(vertices_half[:,0])
        vertices_half[:,1] -= y_max
        full_vertices = torch.vstack((vertices_half,-vertices_half[1:] + vertices_half[0]+ vertices_half[-1])) + c
        return full_vertices

    def polytope2(self):
        '''
        converts a zonotope from a G- to a H- representation
        P
        comb
        isDeg
        '''

        #z = self.deleteZerosGenerators()
        c = self.center
        G = torch.clone(self.generators)
        '''
        h = torch.linalg.vector_norm(G,dim=1)
        h_sort, indicies = torch.sort(h,descending=True)
        h_zero = h_sort < 1e-6
        if torch.any(h_zero):
            first_reduce_idx = torch.nonzero(h_zero)[0,0]
            Gunred = G[indicies[:first_reduce_idx]]
            # Gred = G[indicies[first_reduce_idx:]]
            # d = torch.sum(abs(Gred),0)
            # G = torch.vstack((Gunred,torch.diag(d)))
            G = Gunred
        '''
        n_gens, dim = G.shape
                        
        if dim == 1:
            C = G/torch.linalg.vector_norm(G,dim=1).reshape(-1,1)
        elif dim == 2:      
            C = torch.hstack((-G[:,1],G[:,0]))
            C = C/torch.linalg.vector_norm(C,dim=1).reshape(-1,1)
        elif dim == 3:
            # not complete for example when n_gens < dim-1; n_gens =0 or n_gens =1 
            comb = torch.combinations(torch.arange(n_gens),r=dim-1)
            
            Q = torch.hstack((G[comb[:,0]],G[comb[:,1]]))
            C = torch.hstack((Q[:,1:2]*Q[:,5:6]-Q[:,2:3]*Q[:,4:5],-Q[:,0:1]*Q[:,5:6]-Q[:,2:3]*Q[:,3:4],Q[:,0:1]*Q[:,4:5]-Q[:,1:2]*Q[:,3:4]))
            C = C/torch.linalg.vector_norm(C,dim=1).reshape(-1,1)
        elif dim >=4 and dim<=7:
            assert False
        else:
            assert False
        
        #index = torch.sum(torch.isnan(C),dim=1) == 0
        #C = C[index]
        deltaD = torch.sum(abs(C@G.T),dim=1)
        d = (C@c)
        PA = torch.vstack((C,-C))
        Pb = torch.hstack((d+deltaD,-d+deltaD))
        return PA, Pb, C

    def polytope(self):
        '''
        converts a zonotope from a G- to a H- representation
        self: <zonotope>
        return,
        A: <torch.tensor>, shape [*,nx]
        b: <torch.tensor>, shape [*]

        Ex. 
        if max(A@torch.tensor(nx)-b)>=1e-6:
            NO COLLISION !!!
        else:
            COLLISION !!!


        '''
        c = self.center
        G = torch.clone(self.generators)
        h = torch.linalg.vector_norm(G,dim=1)
        h_sort, indicies = torch.sort(h,descending=True)
        h_zero = h_sort < 1e-6
        if torch.any(h_zero):
            first_reduce_idx = torch.nonzero(h_zero)[0,0]
            Gunred = G[indicies[:first_reduce_idx]]
            # Gred = G[indicies[first_reduce_idx:]]
            # d = torch.sum(abs(Gred),0)
            # G = torch.vstack((Gunred,torch.diag(d)))
            G = Gunred

        n_gens, dim = G.shape
                        
        if dim == 1:
            C = G/torch.linalg.vector_norm(G,dim=1).reshape(-1,1)
        elif dim == 2:
            C = torch.hstack((-G[:,1:2],G[:,0:1]))
            C = C/torch.linalg.vector_norm(C,dim=1).reshape(-1,1)
        elif dim == 3:
            # not complete for example when n_gens < dim-1; n_gens =0 or n_gens =1 
            comb = torch.combinations(torch.arange(n_gens),r=dim-1)
            
            Q = torch.hstack((G[comb[:,0]],G[comb[:,1]]))
            C = torch.hstack((Q[:,1:2]*Q[:,5:6]-Q[:,2:3]*Q[:,4:5],-Q[:,0:1]*Q[:,5:6]-Q[:,2:3]*Q[:,3:4],Q[:,0:1]*Q[:,4:5]-Q[:,1:2]*Q[:,3:4]))
            C = C/torch.linalg.vector_norm(C,dim=1).reshape(-1,1)
        elif dim >=4 and dim<=7:
            assert False
        else:
            assert False
        
        index = torch.sum(torch.isnan(C),dim=1) == 0
        C = C[index]
        deltaD = torch.sum(abs(C@G.T),dim=1)
        d = (C@c)
        PA = torch.vstack((C,-C))
        Pb = torch.hstack((d+deltaD,-d+deltaD))
        return PA, Pb
        '''
        dim, n_gens = G.shape
        if torch.matrix_rank(G) >= dim:
            if dim > 1:
                comb = torch.combinations(torch.arange(n_gens,device=self.__device),r=dim-1)
                n_comb = len(comb)
                C = torch.zeros(n_comb,dim, device=self.__device)
                for i in range(n_comb):
                    indices = comb[i,:]
                    Q = G[:,indices]
                    v = ndimCross(Q)
                    C[i,:] = v/torch.linalg.norm(v)
                # remove None rows dues to rank deficiency
                index = torch.sum(torch.isnan(C),axis=1) == 0
                C = C[index,:]
            else: 
                C =torch.eye(1,device=self.__device)

            # build d vector and determine delta d
            deltaD = torch.zeros(len(C),device=self.__device)
            for iGen in range(n_gens):
                deltaD += abs(C@G[:,iGen])
            # compute dPos, dNeg
            dPos, dNeg = C@c + deltaD, - C@c + deltaD
            # construct the overall inequality constraints
            C = torch.hstack((C,-C))
            d = torch.hstack((dPos,dNeg))
            # catch the case where the zonotope is not full-dimensional
            temp = torch.min(torch.sum(abs(C-C[0]),1),torch.sum(abs(C+C[0]),1))
            if dim > 1 and (C.numel() == 0 or torch.all(temp<1e-12) or torch.all(torch.isnan(C)) or torch.any(torch.max(abs(C),0).values<1e-12)):
                S,V,_ = torch.linalg.svd(G)

                Z_ = S.T@torch.hstack((c,G))

                ind = V <= 1e-12

                # 1:len(V) 

                
        return P, comb, isDeg
        '''
    def deleteZerosGenerators(self,eps=0):
        '''
        delete zero vector generators
        self: <zonotope>

        return <zonotope>
        '''
        non_zero_idxs = torch.any(abs(self.generators)>eps,axis=1)
        Z = torch.vstack((self.center,self.generators[non_zero_idxs]))
        return zonotope(Z)

    def polygon_patch(self, alpha = .5, facecolor='none',edgecolor='green',linewidth=.2,dim=[0,1]):
        z = self.project(dim)
        p = z.polygon().to(device='cpu')
        return patches.Polygon(p,alpha=alpha,edgecolor=edgecolor,facecolor=facecolor,linewidth=linewidth)

    def plot(self, ax,facecolor='none',edgecolor='green',linewidth=.2,dim=[0,1]):
        '''
        plot 2 dimensional projection of a zonotope
        self: <zonotope>
        ax: <Axes> axes oject of a figure to plot
        facecolor: <string> color of face
        edgecolor: <string> color of edges

        ex.
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        zono.plot(ax)
        plt.show()
        '''
        
        z = self.project(dim)
        p = z.polygon().to(device='cpu')

        return ax.add_patch(patches.Polygon(p,alpha=.5,edgecolor=edgecolor,facecolor=facecolor,linewidth=linewidth))

    def reduce(self,order,option='girard'):
        if option == 'girard':
            Z = self.deleteZerosGenerators()
            if order == 1:
                center, G = Z.center,Z.generators
                d = torch.sum(abs(G),0)
                Gbox = torch.diag(d)
                ZRed = torch.vstack((center,Gbox))
            else:
                center, Gunred, Gred = pickedGenerators(Z.center,Z.generators,order)
                d = torch.sum(abs(Gred),0)
                Gbox = torch.diag(d)
                ZRed = torch.vstack((center,Gunred,Gbox))
            return zonotope(ZRed)
        else:
            assert False, 'Invalid reduction option'

    def to_polyZonotope(self,dim=None,prop='None'):
        '''
        convert zonotope to polynomial zonotope
        self: <zonotope>
        dim: <int>, dimension to take as sliceable
        return <polyZonotope>
        '''
        if dim is None:
            return polyZonotope(self.Z,0)
        assert isinstance(dim,int) and dim <= self.dimension
        idx = self.generators[:,dim] == 0
        assert sum(~idx) == 1, 'sliceable generator should be one for the dimension.'
        Z = torch.vstack((self.center,self.generators[~idx],self.generators[idx]))
        return polyZonotope(Z,1,prop=prop)

    def to_interval(self):
        c = self.center
        delta = torch.sum(abs(self.Z),dim=0) - abs(c)
        leftLimit, rightLimit = c -delta, c + delta
        return interval(leftLimit,rightLimit)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()
    z = zonotope([[0,0],[1,0],[0,1],[1,1]])
    z.plot(ax)
    plt.autoscale()
    plt.show()

    Z = torch.tensor([[0., 1, 0,1],[0, 0, -1,1],[0,0,0,1]])
    z = zonotope(Z.T)
    print(torch.eye(3)@z)
    print(z-torch.tensor([1,2,3]))
    print(z.Z)
    print(z.slice(2,1).Z)
    print(z)

    #fig = plt.figure()    
   #ax = fig.gca() 
    #z.plot2d(ax)

  
    Z1 = torch.tensor([[0, 1, 0,1,3,4,5,6,7,1,4,4,15,6,1,3],[0, 0, -1,14,5,1,6,7,1,4,33,15,1,2,33,3]])*0.0001
    z1 = zonotope(Z1)
    #z1.plot2d(ax)
    #plt.axis([-5,5,-5,5])
    #plt.show()

