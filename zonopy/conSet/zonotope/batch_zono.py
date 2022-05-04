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
from zonopy.conSet.zonotope.zono import zonotope
class batchZonotope:
    '''
    b-zono: <batchZonotope>, <torch.float64>

    Z: <torch.Tensor> batch of center vector and generator matrix Z = [C,G]
    , shape [B1, B2, .. , Bb, N+1, nx]
    center: <torch.Tensor> center vector
    , shape [B1, B2, .. , Bb, nx] 
    generators: <torch.Tensor> generator matrix
    , shape [B1, B2, .. , Bb, N, nx]
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
        assert len(Z.shape) > 2, f'The dimension of Z input should be either 1 or 2, not {len(Z.shape)}.'
        ############################################################## 
        self.Z = Z
        self.batch_dim = len(Z.shape) - 2
        self.batch_idx_all = tuple([slice(None) for _ in range(self.batch_dim)])
    def __getitem__(self,idx):
        Z = self.Z[idx]
        if len(Z.shape) > 2:
            return batchZonotope(Z)
        else:
            return zonotope(Z)
    # index_select
    @property 
    def batch_shape(self):
        return self.Z.shape[:self.batch_dim]
    @property
    def dtype(self):
        return self.Z.dtype
    @property
    def device(self):
        return self.Z.device
    @property
    def center(self):
        return self.Z[self.batch_idx_all+(0,)]
    @center.setter
    def center(self,value):
        self.Z[self.batch_idx_all+(0,)] = value
    @property
    def generators(self):
        return self.Z[self.batch_idx_all+(slice(1,None),)]
    @generators.setter
    def generators(self,value):
        self.Z[self.batch_idx_all+(slice(1,None),)] = value
    @property 
    def shape(self):
        return (self.Z.shape[-1],)
    @property
    def dimension(self):
        return self.Z.shape[-1]
    @property
    def n_generators(self):
        return self.Z.shape[-2]-1
    def to(self,dtype=None,device=None):    
        Z = self.Z.to(dtype=dtype, device=device)
        return batchZonotope(Z)
    
    def __str__(self):
        zono_str = f"""center: \n{self.center} \n\nnumber of generators: {self.n_generators} 
            \ngenerators: \n{self.generators} \n\ndimension: {self.dimension}\ndtype: {self.dtype} \ndevice: {self.device}"""
        del_dict = {'tensor':' ','    ':' ','(':'',')':''}
        for del_el in del_dict.keys():
            zono_str = zono_str.replace(del_el,del_dict[del_el])
        return zono_str
    def __repr__(self):
        return str(self.Z).replace('tensor','batchZonotope')
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
            Z[self.batch_idx_all+(0,)] += other
        elif isinstance(other, zonotope): 
            assert self.dimension == other.dimension, f'zonotope dimension does not match: {self.dimension} and {other.dimension}.'
            Z = torch.cat(((self.center + other.center).unsqueeze(self.batch_dim),self.generators,other.generators.repeat(self.batch_shape+(1,1,))),self.batch_dim)
        elif isinstance(other, batchZonotope): 
            assert self.dimension == other.dimension, f'zonotope dimension does not match: {self.dimension} and {other.dimension}.'
            Z = torch.cat(((self.center + other.center).unsqueeze(self.batch_dim),self.generators,other.generators),self.batch_dim)
        else:
            assert False, f'the other object is neither a zonotope nor a torch tensor, not {type(other)}.'
        return batchZonotope(Z)

    __radd__ = __add__
    def __sub__(self,other):
        if isinstance(other, torch.Tensor):
            Z = torch.clone(self.Z)
            assert other.shape == self.shape, f'array dimension does not match: should be {self.shape}, not {other.shape}.'
            Z[self.batch_idx_all+(0,)] -= other
        elif isinstance(other, zonotope): 
            assert self.dimension == other.dimension, f'zonotope dimension does not match: {self.dimension} and {other.dimension}.'
            Z = torch.cat(((self.center - other.center).unsqueeze(self.batch_dim),self.generators,other.generators.repeat(self.batch_shape+(1,1,))),self.batch_dim)
        elif isinstance(other, batchZonotope): 
            assert self.dimension == other.dimension, f'zonotope dimension does not match: {self.dimension} and {other.dimension}.'
            Z = torch.cat(((self.center - other.center).unsqueeze(self.batch_dim),self.generators,other.generators),self.batch_dim)
        else:
            assert False, f'the other object is neither a zonotope nor a torch tensor, not {type(other)}.'
        return batchZonotope(Z)
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
        Z[self.batch_idx_all+(slice(1,None),)] = self.Z[self.batch_idx_all+(slice(1,None),)]
        return batchZonotope(Z)    
    
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
        return batchZonotope(Z)

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

        N = len(slice_dim)
        slice_dim, ind = torch.sort(slice_dim)
        slice_pt = slice_pt[ind]

        c = self.center
        G = self.generators

        non_zero_idx = G[self.batch_idx_all+(slice(None),slice_dim)] != 0
        assert torch.all(torch.sum(non_zero_idx,-2)==1), 'There should be one generator for each slice index.'
        slice_idx = torch.any(non_zero_idx!=0,-1)

        slice_c = c[self.batch_idx_all+(slice_dim,)]
        slice_g = G[slice_idx][self.batch_idx_all+(slice_dim,)]


        non_zero_idx = G[:,slice_dim] != 0
        assert torch.all(torch.sum(non_zero_idx,0)==1), 'There should be one generator for each slice index.'
        slice_idx = torch.any(non_zero_idx!=0,1)

        slice_c = c[slice_dim]
        slice_g = G[slice_idx,slice_dim]
        slice_lambda = (slice_pt-slice_c)/slice_g
        assert not any(abs(slice_lambda)>1), 'slice point is ouside bounds of reach set, and therefore is not verified'
        
        Z = torch.vstack((c + slice_lambda@G[slice_idx],G[~slice_idx]))
        return zonotope(Z)

    def project(self,dim=[0,1]):
        '''
        the projection of a zonotope onto the specified dimensions
        self: <zonotope>
        dim: <int> or <list> or <torch.Tensor> dimensions for prjection 
        
        return <zonotope>
        '''
        Z = self.Z[self.batch_idx_all+(slice(None),dim)]
        return batchZonotope(Z)

    def polygon(self):
        '''
        converts a 2-d zonotope into a polygon as vertices
        self: <zonotope>

        return <torch.Tensor>, <torch.float64>
        '''
        dim = 2
        #z = self.deleteZerosGenerators()
        z =self
        c = z.center.unsqueeze(-2).repeat((1,)*(self.batch_dim+2))
        G = torch.clone(z.generators)
        x_idx = self.batch_idx_all+(slice(None),0)
        y_idx = self.batch_idx_all+(slice(None),1)
        G_y = G[y_idx]
        x_max = torch.sum(abs(G[x_idx]),-1)
        y_max = torch.sum(abs(G_y),-1)
        G[G_y<0] = - G[G_y<0]
        angles = torch.atan2(G[x_idx],G[y_idx])
        #ang_idx = torch.argsort(angles,dim=-1)
        ang_idx = torch.argsort(angles,dim=-1).unsqueeze(-1).repeat((1,)*(self.batch_dim+1)+self.shape)
        vertices_half = torch.cat((torch.zeros(self.batch_shape+(1,)+self.shape),2*G.gather(-2,ang_idx).cumsum(axis=self.batch_dim)),-2)
        vertices_half[x_idx] += (x_max - torch.max(vertices_half[x_idx],dim=-1)[0]).unsqueeze(-1)
        vertices_half[y_idx] -= y_max.unsqueeze(-1)
        
        
        temp = (vertices_half[self.batch_idx_all+(0,)]+ vertices_half[self.batch_idx_all+(-1,)]).unsqueeze(-2)
        full_vertices = torch.cat((vertices_half,-vertices_half[self.batch_idx_all+(slice(1,None),)] + temp),dim=self.batch_dim) + c
        return full_vertices        

    def polytope(self):
        '''
        converts a zonotope from a G- to a H- representation
        P
        comb
        isDeg
        '''

        #z = self.deleteZerosGenerators()
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
        
        index = torch.sum(torch.isnan(C),dim=1) == 0
        C = C[index]
        deltaD = torch.sum(abs(C@G.T),dim=1)
        d = (C@c)
        PA = torch.vstack((C,-C))
        Pb = torch.hstack((d+deltaD,-d+deltaD))
        return PA, Pb, C

    def deleteZerosGenerators(self,sorted=False,sort=False):
        '''
        delete zero vector generators
        self: <zonotope>

        return <zonotope>
        '''
        if sorted:
            non_zero_idxs = torch.sum(torch.any(self.generators!=0,-1),tuple(range(self.batch_dim))) != 0
            g_red = self.generators[self.batch_idx_all+(non_zero_idxs,)]
        else: 
            zero_idxs = torch.any(self.generators==0,axis=-1)
            ind = zero_idxs.sort(-1)[1].unsqueeze(-1).repeat((1,)*(self.batch_dim+1)+self.shape)
            max_non_zero_len = (~zero_idxs).sum(-1).max()
            g_red = self.generators.gather(-2,ind)[self.batch_idx_all+(slice(None,max_non_zero_len),)]
        Z = torch.cat((self.center.unsqueeze(self.batch_dim),g_red),self.batch_dim)
        return batchZonotope(Z)


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
        
        Z = self.project(dim)
        P = Z.polygon().to(device='cpu')
        P = P.reshape(torch.prod(torch.tensor(self.batch_shape)),-1,self.dimension)
        for i in range(len(P)):
            ax.add_patch(patches.Polygon(P[i],alpha=.5,edgecolor=edgecolor,facecolor=facecolor,linewidth=linewidth))

    def reduce(self,order,option='girard'):
        if option == 'girard':
            Z = self.deleteZerosGenerators()
            center, Gunred, Gred = pickedGenerators(Z.center,Z.generators,order)
            d = torch.sum(abs(Gred),self.batch_dim)
            Gbox = torch.diag_embed(d)
            ZRed= torch.cat((center.unsqueeze(self.batch_dim),Gunred,Gbox),self.batch_dim)
            return batchZonotope(ZRed)
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
        delta = torch.sum(abs(self.Z),self.batch_dim) - abs(c)
        leftLimit, rightLimit = c -delta, c + delta
        return interval(leftLimit,rightLimit)
    
    
    
    '''
    c = self.center
    g = self.generators.sort(self.batch_dim,descending=True)[0]


    '''
    def pickedGenerators(self,order):
        '''
        selects generators to be reduced
        '''
        c = self.center
        G = self.generators.sort(self.batch_dim,descending=True)[0]
        
        dim = c.shape
        norm_dim = tuple(range(1,len(dim)+1))
        nrOfGens = self.n_generators
        if nrOfGens != 0:
            d = self.dimension
            # only reduce if zonotope order is greater than the desired order
            if nrOfGens > d*order:
                
                # compute metric of generators
                h = torch.linalg.vector_norm(G,1,-1) - torch.linalg.vector_norm(G,torch.inf,-1) #NOTE: -1

                # number of generators that are not reduced
                nUnreduced = int(d*(order-1))
                nReduced = nrOfGens - nUnreduced 
                # pick generators with smallest h values to be reduced
                sorted_h = torch.argsort(h,-1).unsqueeze(-1).repeat((1,)*(self.batch_dim+1)+self.shape) #NOTE: -1
                Gsorted = G.gather(self.batch_dim,sorted_h)
                Gred = Gsorted[self.batch_idx_all+(slice(None,nReduced),)]
                Gunred = Gsorted[self.batch_idx_all+(slice(nReduced,None),)]
            else:
                Gred = torch.tensor([]).reshape(self.batch_shape+(0,)+self.shape)
                Gunred = G
        else:
            Gred = torch.tensor([]).reshape(self.batch_shape+(0,)+self.shape)
            Gunred = torch.tensor([]).reshape(self.batch_shape+(0,)+self.shape)

        return c, Gunred, Gred


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time 
    n_gens = 100
    ZZ = torch.rand(n_gens,3)
    N_z = (100,1)
    t_s = time.time()    
    Z = ZZ.repeat(N_z+(1,1))
    Z[:,:,:,0] += torch.arange(torch.prod(torch.tensor(N_z))).reshape(N_z+(1,))
    Z = batchZonotope(Z)
    Z2 = torch.eye(3)@Z
    print(time.time()-t_s)

    t_s = time.time()  
    for i in range(N_z[0]):
        for j in range(N_z[1]):
            Z = ZZ
            Z[0] += torch.tensor([j+N_z[1]*i])
            Z = zonotope(Z)
            Z2 = torch.eye(3)@Z
    print(time.time()-t_s)

    '''
    fig = plt.figure()
    ax = fig.gca()
    t_s = time.time()    
    Z = torch.tensor([[0.0,0],[1,0],[0,1]]).repeat(N_z,1,1)
    Z[:,0] += torch.arange(N_z).reshape(-1,1)
    Z = batchZonotope(Z)
    Z.plot(ax)
    print(time.time()-t_s)
    plt.autoscale()
    plt.show()

    fig = plt.figure()
    ax = fig.gca()
    t_s = time.time()    
    for i in range(N_z):
        Z = torch.tensor([[0.0,0],[1,0],[0,1]])
        Z[0] += torch.tensor([i])
        Z = zonotope(Z)
        Z.plot(ax)
    print(time.time()-t_s)
    plt.autoscale()
    plt.show()
    '''