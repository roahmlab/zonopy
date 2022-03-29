"""
Define class for zonotope
Reference: CORA
Writer: Yongseok Kwon
"""


import torch
import matplotlib.patches as patches
from zonopy.conSet import DEFAULT_DTYPE, DEFAULT_DEVICE
from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope 
from zonopy.conSet.utils import delete_column
from zonopy.conSet.zonotope.utils import pickedGenerators
class zonotope:
    '''
    zono: <zonotope>, <torch.float64>

    Z: <torch.Tensor> center vector and generator matrix Z = [c,G]
    , shape [nx, N+1] OR [nx], where N = 0
    -> shape [nx, N+1]
    dtype: data type of class properties
    , torch.float or torch.double
    device: device for torch
    , 'cpu', 'gpu', 'cuda', ...
    center: <torch.Tensor> center vector
    , shape [nx,1] 
    generators: <torch.Tensor> generator matrix
    , shape [nx, N]
        
    
    Eq. (coeff. a1,a2,...,aN \in [0,1])
    G = [g1,g2,...,gN]
    zono = c + a1*g1 + a2*g2 + ... + aN*gN
    '''
    def __init__(self,Z,dtype=DEFAULT_DTYPE,device=DEFAULT_DEVICE):
        if type(Z) == list:
            Z = torch.tensor(Z)
        assert type(Z) == torch.Tensor, f'The input matrix should be either torch tensor or list, but {type(Z)}.'
        if len(Z.shape) == 1:
            Z = Z.reshape(1,-1)
        if dtype == float:
            dtype = torch.double
        assert dtype == torch.float or dtype == torch.double, f'dtype should be either torch.float (torch.float32) or torch.double (torch.float64), but {dtype}.'
        assert len(Z.shape) == 2, f'The dimension of Z input should be either 1 or 2, but {len(Z.shape)}.'

        self.dtype = dtype
        self.device = device
        self.Z = Z.to(dtype=dtype,device=device)
        self.center = self.Z[:,0]
        self.generators = self.Z[:,1:]
        self.dimension = self.Z.shape[0]
    @property
    def n_generators(self):
        return self.generators.shape[1]

    def __str__(self):
        zono_str = f"""center: \n{self.center.to(dtype=torch.float,device='cpu')} \n\nnumber of generators: {self.n_generators} 
            \ngenerators: \n{self.generators.to(dtype=torch.float,device='cpu')} \n\ndimension: {self.dimension}\ndtype: {self.dtype} \ndevice: {self.device}"""
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
        if type(other) == torch.Tensor:
            Z = torch.clone(self.Z)
            assert other.shape == Z[:,0].shape, f'array dimension does not match: should be {Z[:,0].shape}, but {other.shape}.'
            Z[:,0] += other
                
        elif type(other) == zonotope: 
            assert self.dimension == other.dimension, f'zonotope dimension does not match: {self.dimension} and {other.dimension}.'
            Z = torch.hstack([self.center + other.center,self.generators,other.generators])

        else:
            raise ValueError(f'the other object is neither a zonotope nor a torch tensor: {type(other)}.')

        return zonotope(Z,self.dtype,self.device)

    __radd__ = __add__
    def __sub__(self,other):
        return self.__add__(-other)
    def __rsub__(self,other):
        return -self.__sub__(other)
    # TODO: __iadd__
    def __pos__(self):
        return self    
    
    def __neg__(self):
        '''
        Overloaded unary '-' operator for negation
        self: <zonotope>
        return <zonotope>
        '''   
        Z = -self.Z
        Z[:,1:] = self.Z[:,1:]
        return zonotope(Z,self.dtype,self.device)    
    
    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for matrix multiplication
        self: <zonotope>
        other: <torch.tensor>
        
        zono = other @ self

        return <zonotope>
        '''   
        assert type(other) == torch.Tensor, f'the other object should be torch tensor, but {type(other)}.'
        other = other.to(dtype=self.dtype)
        Z = other @ self.Z
        return zonotope(Z,self.dtype,self.device)

    def __matmul__(self,other):
        assert type(other) == torch.Tensor, f'the other object should be torch tensor, but {type(other)}.'
        other = other.to(dtype=self.dtype)
        Z = self.Z @ other
        return zonotope(Z,self.dtype,self.device)   
        
    def slice(self,slice_dim,slice_pt):
        '''
        
        slice zonotope on specified point in a certain dimension
        self: <zonotope>
        slice_dim: <torch.Tensor> or <list> or <int>
        , shape  []
        slice_pt: <torch.Tensor> or <list> or <float> or <int>
        , shape  []
        NOTE: latter allow slice_pt to be interval

        return <zonotope>
        '''
        if type(slice_dim) == list:
            slice_dim = torch.tensor(slice_dim,dtype=int,device=self.device)
        elif type(slice_dim) == int or (type(slice_dim) == torch.Tensor and len(slice_dim.shape)==0):
            slice_dim = torch.tensor([slice_dim],dtype=int,device=self.device)

        if type(slice_pt) == list:
            slice_pt = torch.tensor(slice_pt,dtype=self.dtype,device=self.device)
        elif type(slice_pt) == int or type(slice_pt) == float or (type(slice_pt) == torch.Tensor and len(slice_pt.shape)==0):
            slice_pt = torch.tensor([slice_pt],dtype=self.dtype,device=self.device)

        assert type(slice_dim) == torch.Tensor and type(slice_pt) == torch.Tensor, 'Wrong type of input'
        assert len(slice_dim.shape) ==1, 'slicing dimension should be 1-dim component.'
        assert len(slice_pt.shape) ==1, 'slicing point should be 1-dim component.'
        assert len(slice_dim) == len(slice_pt), f'The number of slicing dimension ({len(slice_dim)}) and the number of slicing point ({len(slice_dim)}) should be the same.'

        N = len(slice_dim)
        
        Z = self.Z
        c = self.center
        G = self.generators

        slice_idx = torch.zeros(N,dtype=int,device=self.device)
        for i in range(N):
            non_zero_idx = (G[slice_dim[i],:] != 0).nonzero().reshape(-1)
            if len(non_zero_idx) != 1:
                if len(non_zero_idx) == 0:
                    raise ValueError('no generator for slice index')
                else:
                    raise ValueError('more than one generators for slice index')
            slice_idx[i] = non_zero_idx

        slice_c = c[slice_dim]

        slice_G = torch.zeros(N,N,dtype=self.dtype,device=self.device)
        for i in range(N):
            slice_G[i] = G[slice_dim[i],slice_idx]
        
        slice_lambda = torch.linalg.solve(slice_G, slice_pt - slice_c)

        assert not any(abs(slice_lambda)>1), 'slice point is ouside bounds of reach set, and therefore is not verified'

        Z_new = torch.zeros(Z.shape,dtype=self.dtype,device=self.device)
        Z_new[:,0] = c + G[:,slice_idx]@slice_lambda
        Z_new[:,1:] = G
        Z_new = delete_column(Z_new,slice_idx+1)
        return zonotope(Z_new,self.dtype,self.device)

    def project(self,dim=[0,1]):
        '''
        the projection of a zonotope onto the specified dimensions
        self: <zonotope>
        dim: <int> or <list> or <torch.Tensor> dimensions for prjection 
        
        return <zonotope>
        '''
        Z = self.Z[dim,:]
        return zonotope(Z,self.dtype,self.device)

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
        x_max = torch.sum(abs(G[0,:]))
        y_max = torch.sum(abs(G[1,:]))
        G[:,z.generators[1,:]<0] = - z.generators[:,z.generators[1,:]<0] # make all y components as positive
        angles = torch.atan2(G[1,:], G[0,:])
        ang_idx = torch.argsort(angles)
        
        vertices_half = torch.zeros(dim,n+1,dtype=self.dtype,device=self.device)
        for i in range(n):
            vertices_half[:,i+1] = vertices_half[:,i] + 2*G[:,ang_idx[i]]
        
        vertices_half[0,:] += (x_max-torch.max(vertices_half[0,:]))
        vertices_half[1,:] -= y_max

        full_vertices = torch.zeros(dim,2*n+1,dtype=self.dtype,device=self.device)
        full_vertices[:,:n+1] = vertices_half
        full_vertices[:,n+1:] = -vertices_half[:,1:] + vertices_half[:,0].reshape(dim,1) + vertices_half[:,-1].reshape(dim,1) #flipped
        
        full_vertices += c.reshape(dim,1)
        return full_vertices.to(dtype=self.dtype,device=self.device)

    def deleteZerosGenerators(self):
        '''
        delete zero vector generators
        self: <zonotope>

        return <zonotope>
        '''
        non_zero_idxs = torch.any(self.generators!=0,axis=0)
        Z_new = self.Z[:,[i for i in range(self.n_generators+1) if i==0 or non_zero_idxs[i-1]]]
        return zonotope(Z_new,dtype=self.dtype,device=self.device)

    def plot(self, ax,facecolor='none',edgecolor='green',linewidth=.2):
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
        zono.plot2d(ax)
        plt.show()
        '''
        

        dim = 2
        z = self.project(torch.arange(dim))
        p = z.polygon()

        ax.add_patch(patches.Polygon(p.T,alpha=.5,edgecolor=edgecolor,facecolor=facecolor,linewidth=linewidth,))

    def to_polyZonotope(self,dim=None,):
        '''
        convert zonotope to polynomial zonotope
        self: <zonotope>
        dim: <int>, dimension to take as sliceable
        return <polyZonotope>
        '''
        if dim is None:
            return polyZonotope(self.center,Grest = self.generators)
        assert type(dim) == int
        assert dim <= self.dimension

        g_row_dim =self.generators[dim,:]
        idx = (g_row_dim!=0).nonzero().reshape(-1)
        
        assert idx.numel() != 0, 'no sliceable generator for the dimension.'
        assert idx.numel() == 1,'more than one no sliceable generators for the dimesion.'        
        
        c = self.center
        G = self.generators[:,idx]
        Grest = delete_column(self.generators,idx)

        return polyZonotope(c,G,Grest)

    def reduce(self,order,option='girard'):
        if option == 'girard':
            center, Gunred, Gred = pickedGenerators(self,order)
            d = torch.sum(abs(Gred),1)
            Gbox = torch.diag(d)
            ZRed = torch.hstack((center.reshape(-1,1),Gunred,Gbox))
            return zonotope(ZRed,self.dtype,self.device)
        else:
            assert False, 'Invalid reduction method'

if __name__ == '__main__':
    import matplotlib.pyplot as plt
        
    Z = torch.tensor([[0, 1, 0,1],[0, 0, -1,1],[0,0,0,1]])
    z = zonotope(Z)
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

