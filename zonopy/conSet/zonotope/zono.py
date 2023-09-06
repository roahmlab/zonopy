"""
Define class for zonotope
Author: Yongseok Kwon
Reference: CORA
"""
import torch
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope
from zonopy.conSet.interval.interval import interval
from zonopy.conSet.zonotope.utils import pickedGenerators, ndimCross
from scipy.spatial import ConvexHull
from ..gen_ops import (
    _add_genzono_impl,
    _add_genzono_num_impl,
    _mul_genzono_num_impl,
    )


class zonotope:
    '''
    zono: <zonotope>

    Z: <torch.Tensor> center vector and generator matrix Z = [c,G]
    , shape [N+1, nx]
    center: <torch.Tensor> center vector
    , shape [nx] 
    generators: <torch.Tensor> generator matrix
    , shape [N, nx]
    
    Eq.
    G = [[g1],[g2],...,[gN]]
    zono = {c + a1*g1 + a2*g2 + ... + aN*gN | coeff. a1,a2,...,aN \in [-1,1] }
    '''
    def __init__(self,Z, dtype=None, device=None):
        ################ may not need these for speed ################ 
        # Make sure Z is a tensor
        if not isinstance(Z, torch.Tensor) and dtype is None:
            dtype = torch.get_default_dtype()
        Z = torch.as_tensor(Z, dtype=dtype, device=device)
        
        assert len(Z.shape) == 2, f'The dimension of Z input should be either 1 or 2, not {len(Z.shape)}.'
        ############################################################## 
        self.Z = Z
    @property
    def dtype(self):
        '''
        The data type of a zonotope properties
        return torch.float or torch.double        
        '''
        return self.Z.dtype
    @property
    def device(self):
        '''
        The device of a zonotope properties
        return 'cpu', 'cuda:0', or ...
        '''
        return self.Z.device
    @property
    def center(self):
        '''
        The center of a zonotope
        return <torch.Tensor>
        , shape [nx] 
        '''
        return self.Z[0]
    @center.setter
    def center(self,value):
        '''
        Set value of the center
        '''
        self.Z[0] = value
    @property
    def generators(self):
        '''
        Generators of a zonotope
        return <torch.Tensor>
        , shape [N, nx]
        '''
        return self.Z[1:]
    @generators.setter
    def generators(self,value):
        '''
        Set value of generators
        '''
        self.Z[1:] = value
    @property 
    def shape(self):
        '''
        The shape of vector elements (ex. center) of a zonotope
        return <tuple>, (nx,)
        '''
        return (self.Z.shape[1],)
    @property
    def dimension(self):
        '''
        The dimension of a zonotope
        return <int>, nx
        '''        
        return self.Z.shape[1]
    @property
    def n_generators(self):
        '''
        The number of generators of a zonotope
        return <int>, N
        '''
        return len(self.Z)-1
    def to(self,dtype=None,device=None):
        '''
        Change the device and data type of a zonotope
        dtype: torch.float or torch.double
        device: 'cpu', 'gpu', 'cuda:0', ...
        '''
        Z = self.Z.to(dtype=dtype, device=device, non_blocking=True)
        return zonotope(Z)
    def cpu(self):
        '''
        Change the device of a zonotope to CPU
        '''
        Z = self.Z.cpu()
        return zonotope(Z)

    def __repr__(self):
        '''
        Representation of a zonotope as a text
        return <str>, 
        ex. zonotope([[0., 0., 0.],[1., 0., 0.]])
        '''
        return str(self.Z).replace('tensor','zonotope')
    def  __add__(self,other):
        '''
        Overloaded '+' operator for addition or Minkowski sum
        self: <zonotope>
        other: <torch.Tensor> OR <zonotope>
        return <zonotope>
        '''   
        if isinstance(other, (torch.Tensor, float, int)):
            Z = _add_genzono_num_impl(self, other)
            return zonotope(Z)
        
        elif isinstance(other, zonotope): 
            Z = _add_genzono_impl(self, other)
            return zonotope(Z)
        
        else:
            return NotImplemented

    __radd__ = __add__ # '+' operator is commutative.

    def __sub__(self,other):
        '''
        Overloaded '-' operator for substraction or Minkowski difference
        self: <zonotope>
        other: <torch.Tensor> OR <zonotope>
        return <zonotope>        
        '''
        import warnings
        warnings.warn(
            "PZ subtraction as addition of negative is deprecated and will be removed to reduce confusion!",
            DeprecationWarning)
        return self.__add__(-other)
    def __rsub__(self,other): 
        '''
        Overloaded reverted '-' operator for substraction or Minkowski difference
        self: <zonotope>
        other: <torch.Tensor> OR <zonotope>
        return <zonotope>                
        '''
        import warnings
        warnings.warn(
            "PZ subtraction as addition of negative is deprecated and will be removed to reduce confusion!",
            DeprecationWarning)
        return -self.__sub__(other)
    def __pos__(self):
        '''
        Overloaded unary '+' operator for a zonotope ifself
        self: <zonotope>
        return <zonotope>
        '''   
        return self    
    
    def __neg__(self):
        '''
        Overloaded unary '-' operator for negation of a zonotope
        self: <zonotope>
        return <zonotope>
        '''   
        Z = torch.clone(self.Z)
        Z[0] *= -1
        return zonotope(Z)    
    
    def __rmatmul__(self,other):
        '''
        Overloaded reverted '@' operator for matrix multiplication on vector elements of a zonotope
        self: <zonotope>
        other: <torch.Tensor>
        return <zonotope>
        '''   
        assert isinstance(other, torch.Tensor), f'The other object should be torch tensor, but {type(other)}.'
        Z = self.Z@other.T
        return zonotope(Z)
    def __mul__(self,other):
        '''
        Overloaded reverted '*' operator for scaling a zonotope
        self: <zonotope>
        other: <int> or <float>
        return <zonotope>
        '''   
        if isinstance(other,(torch.Tensor,int,float)):
            Z = _mul_genzono_num_impl(self, other)
            return zonotope(Z)
        
        else:
            return NotImplemented

    __rmul__ = __mul__ # '*' operator is commutative.

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
            slice_dim = torch.tensor(slice_dim,dtype=torch.long,device=self.device)
        elif isinstance(slice_dim, int) or (isinstance(slice_dim, torch.Tensor) and len(slice_dim.shape)==0):
            slice_dim = torch.tensor([slice_dim],dtype=torch.long,device=self.device)

        if isinstance(slice_pt, list):
            slice_pt = torch.tensor(slice_pt,dtype=self.dtype,device=self.device)
        elif isinstance(slice_pt, int) or isinstance(slice_pt, float) or (isinstance(slice_pt, torch.Tensor) and len(slice_pt.shape)==0):
            slice_pt = torch.tensor([slice_pt],dtype=self.dtype,device=self.device)

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
        The projection of a zonotope onto the specified dimensions
        self: <zonotope>
        dim: <int> or <list> or <torch.Tensor> dimensions for prjection 
        
        return <zonotope>
        '''
        Z = self.Z[:,dim]
        return zonotope(Z)

    def polygon(self):
        '''
        Vertice representation as a polygon from a 2-dimensional zonotope

        return <torch.Tensor>, <torch.float> or <torch.double>
        , shape [P,2], where P is the number of vertices
        '''
        dim = 2
        z = self.deleteZerosGenerators()
        c = z.center[:2]
        G = torch.clone(z.generators[:,:2])
        n = z.n_generators
        x_max = torch.sum(abs(G[:,0]))
        y_max = torch.sum(abs(G[:,1]))
        
        G[z.generators[:,1]<0,:2] = - z.generators[z.generators[:,1]<0,:2] # make all y components as positive
        angles = torch.atan2(G[:,1], G[:,0])
        ang_idx = torch.argsort(angles)
                
        vertices_half = torch.vstack((torch.zeros(dim,device=self.device),2*G[ang_idx].cumsum(axis=0)))
        vertices_half[:,0] += x_max - torch.max(vertices_half[:,0])
        vertices_half[:,1] -= y_max

        full_vertices = torch.vstack((vertices_half,-vertices_half[1:] + vertices_half[0]+ vertices_half[-1])) + c
        return full_vertices
        
    def polyhedron(self):
        '''
        Vertice representation as a polygon from a 3-dimensional zonotope

        return <torch.Tensor>, <torch.float> or <torch.double>
        , shape [P,3], where P is the number of vertices
        '''
        dim = 3
        V = self.center[:dim]
        for i in range(self.n_generators):
            translation = self.Z[i+1,:dim]
            V = torch.vstack((V+translation,V-translation))
            if i > dim:
                try:
                    K = ConvexHull(V)
                    V = V[K.vertices]
                except:
                    V = V
        return V


    def polytope(self,combs=None):
        '''
        Half-plane representation of zonotope
        return,
        A: <torch.tensor>, 
        shape [*,nx]
        b: <torch.tensor>, 
        shape [*]

        A point, x (torch.Tensor, shape (nx,)), is at outside of a zonotope
        <-> max(A@x-b)>=0 (you might wanna use 1e-6 as a threshold instead for numerical stability)

        A point, x, is inside of a zonotope
        <-> max(A@x-b)<0
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
            if combs is None or n_gens >= len(combs):
                comb = torch.combinations(torch.arange(n_gens),r=dim-1)
            else:
                comb = combs[n_gens]
            
            Q = torch.hstack((G[comb[:,0]],G[comb[:,1]]))
            C = torch.hstack((Q[:,1:2]*Q[:,5:6]-Q[:,2:3]*Q[:,4:5],-Q[:,0:1]*Q[:,5:6]+Q[:,2:3]*Q[:,3:4],Q[:,0:1]*Q[:,4:5]-Q[:,1:2]*Q[:,3:4]))
            C = C/torch.linalg.vector_norm(C,dim=1).reshape(-1,1)
        elif dim >=4 and dim<=7:
            assert False
        else:
            assert False
        
        index = torch.sum(torch.isnan(C),dim=1) == 0
        C = C[index]
        deltaD = torch.sum(abs(C@self.generators.T),dim=1)
        d = (C@c)
        PA = torch.vstack((C,-C))
        Pb = torch.hstack((d+deltaD,-d+deltaD))
        return PA, Pb

    def deleteZerosGenerators(self,eps=0):
        '''
        Delete zero vector generators
        return <zonotope>
        '''
        non_zero_idxs = torch.any(abs(self.generators)>eps,axis=1)
        Z = torch.vstack((self.center,self.generators[non_zero_idxs]))
        return zonotope(Z)

    def polygon_patch(self, alpha = .5, facecolor='none',edgecolor='green',linewidth=.2,dim=[0,1]):
        z = self.project(dim)
        p = z.polygon().cpu().detach()
        return patches.Polygon(p,alpha=alpha,edgecolor=edgecolor,facecolor=facecolor,linewidth=linewidth)

    def polyhedron_patch(self,alpha = .5, facecolor='none',edgecolor='green',linewidth=.2):
        dim = 3
        Z = self.Z.cpu().detach()
        V = Z[0,:dim]
        for i in range(self.n_generators):
            translation = Z[i+1,:dim]
            V = torch.vstack((V+translation,V-translation))
            if dim < i < self.n_generators:
                try:
                    K = ConvexHull(V)
                    V = V[K.vertices]
                except:
                    V = V
        K = ConvexHull(V)
        V = V.unsqueeze(0)        
        return torch.cat([V[:,s] for s in K.simplices])
 
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
        p = z.polygon().cpu()

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

    def to_polyZonotope(self,dim=None,id=None):
        '''
        Convert zonotope to polynomial zonotope
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
        return polyZonotope(Z,1,id=id)

    def to_interval(self):
        '''
        Convert zonotope to interval
        return <interval>
        '''
        c = self.center
        delta = torch.sum(abs(self.Z),dim=0) - abs(c)
        leftLimit, rightLimit = c -delta, c + delta
        return interval(leftLimit,rightLimit)

if __name__ == '__main__':
    
    
    import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection="3d")
    import mpl_toolkits.mplot3d as a3
    fig = plt.figure()
    ax = a3.Axes3D(fig)
    z1 = zonotope([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0]])
    z2 = zonotope([[1,2,3],[3,2,4],[3,2,5],[4,5,9],[3,9,4],[8,0,3],[5,6,1],[3,4,5],[1,1,0]])
    
    #ax.dist=30
    #ax.azim=-140
    #ax.elev=20


    #ax.set_xlim([-40,40])
    #ax.set_ylim([-40,40])
    #ax.set_zlim([-40,40])
    
    p1 = z1.polyhedron_patch()
    p2 = z2.polyhedron_patch()
    P = p1+p2
    #patch = Poly3DCollection(z2.polyhedron_patch(),alpha = 0.1, facecolor='green',edgecolor='black')
    
    #ax.add_collection3d(patch)
    patch1 = Poly3DCollection(P,alpha = 0.1, facecolor='red',edgecolor='black')
    ax.add_collection3d(patch1)

    for i, a in enumerate('xyz'):        
        ub = patch1._vec.max(1)[0]
        lb = patch1._vec.min(1)[0]
        getattr(ax, f'set_{a}lim')([lb-10,ub+10])
    #ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    plt.show()

    import pdb;pdb.set_trace()
    
    ##
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

