#%%
# TODO:
# 1. represent zonotope in polytope
# 2. represent zonotope in polygon 3D
# 3. plot 3D

import torch 
from numpy import argwhere as np_argwhere 
# NOTE: recent version (1.10.XX) of torch doesn't provide argwhere function, but upcomming version (1.11.XX) will include this.
from utils import delete_column

import numpy as np


class zonotope:
    '''
    zono: <zonotope>, <torch.float64>

    Z: <torch.Tensor> center vector and generator matrix Z = [c,G]
    , shape: [nx, N+1]
    
    center: <torch.Tensor> center vector
    , shape: [nx,1] 
    generators: <torch.Tensor> generator matrix
    , shape: [nx, N]
    
    
    Eq. (coeff. a1,a2,...,aN ~ [0,1])
    G = [g1,g2,...,gN]
    zono = c + a1*g1 + a2*g2 + ... + aN*gN
    '''
    def __init__(self,Z):
        assert type(Z) == torch.Tensor, f'The input matrix should be torch tensor, but {type(Z)}.'
        Z = Z.to(dtype=torch.float64)
        self.Z = Z
        self.center = self.Z[:,0]
        self.generators = self.Z[:,1:]
        self.dim = self.Z.shape[0]
        self.n_generators = self.Z.shape[1] - 1
    
    def __str__(self):
        return f'\ncenter: \n {self.center} \n\n generators: \n {self.generators} \n\n dimension: {self.dim} \n\n number of generators: {self.n_generators}\n'
    
    def __repr__(self):
        return str(Z).replace('tensor','zonotope')
    
    def  __add__(self,other):
        '''
        Overloaded '+' operator for Minkowski sum
        self: <zonotope>
        other: <torch.tensor> OR <zonotope>
        return <polyZonotope>
        '''   
        if type(other) == torch.Tensor:
            Z = self.Z
            assert other.shape == Z[:,0].shape, f'array dimension does not match: should be {Z[:,0].shape}, but {other.shape}.'
            Z[:,0] += other
                
        elif type(other) == zonotope: # Minkowski sum
            assert self.dim == other.dim, f'zonotope dimension does not match: {self.dim} and {other.dim}.'
            
            Z = torch.zeros(self.dim,1+self.n_generators+other.n_generators)
            Z[:,0] = self.center + other.center
            Z[:,1:1+self.n_generators] = self.generators
            Z[:,1+self.n_generators:] = other.generators

        else:
            raise ValueError(f'the other object is neither a zonotope nor a torch tensor: {type(other)}.')

        return zonotope(Z)
    
    #__radd__ == __add__

    def  __sub__(self,other):
        '''
        Overloaded '-' operator for Minkowski substaction
        self: <zonotope>
        other: <torch.tensor> OR <zonotope>
        return <zonotope>
        '''   
        if type(other) == torch.Tensor:
            Z = self.Z
            assert other.shape == Z[:,0].shape, f'array dimension does not match: should be {Z[:,0].shape}, but {other.shape}.'
            Z[:,0] -= other
                
        elif type(other) == zonotope: # Minkowski sum
            assert self.dim != other.dim, f'zonotope dimension does not match: {self.dim} and {other.dim}.'
            
            Z = torch.zeros(self.dim,1+self.n_generators+other.n_generators)
            Z[:,0] = self.center - other.center
            Z[:,1:1+self.n_generators] = self.generators
            Z[:,1+self.n_generators:] = other.generators
            
        else:
            raise ValueError(f'the other object is neither a zonotope nor an array: {type(other)}.')

        return zonotope(Z)
    
    def __pos__(self):
        '''
        Overloaded unary '+' operator for positive
        self: <zonotope>
        return <zonotope>
        '''   
        return self    
    
    def __neg__(self):
        '''
        Overloaded unary '-' operator for negation
        self: <zonotope>
        return <zonotope>
        '''   
        Z = -self.Z
        Z[:,1:] = self.Z[:,1:]
        return zonotope(Z)    
    
    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for matrix multiplication
        self: <zonotope>
        other: <torch.tensor>
        
        zono = other @ self

        return <zonotope>
        '''   
        assert type(other) == torch.Tensor, f'the other object should be torch tensor, but {type(other)}.'
        other = other.to(dtype=torch.float64)
        Z = other @ self.Z
        return zonotope(Z)

    def slice(self,slice_dim,slice_pt):
        '''
        
        slice zonotope on specified point in a certain dimension
        self: <zonotope>
        slice_dim: <torch.Tensor> or <np.ndarray> or <tuple> or <list>
        , shape  []
        slice_pt: <torch.Tensor> or <np.ndarray> or <tuple> or <list>
        , shape  []
        NOTE: latter allow slice_pt to be interval

        return <zonotope>
        '''
        if type(slice_dim) != torch.Tensor:
            if type(slice_dim) == list or type(slice_dim) == tuple or type(slice_dim) == np.ndarray:
                slice_dim = torch.tensor(slice_dim)
            else:
                slice_dim = torch.tensor([slice_dim])
        if type(slice_pt) != torch.Tensor:
            if type(slice_dim) == list or type(slice_dim) == tuple or type(slice_dim) == np.ndarray:
                slice_pt = torch.tensor(slice_pt)
            else:
                slice_pt = torch.tensor([slice_pt])
        
        assert len(slice_dim.shape) ==1, 'slicing dimension should be 1-dim component.'
        assert len(slice_pt.shape) ==1, 'slicing point should be 1-dim component.'
        assert len(slice_dim) == len(slice_pt), f'The number of slicing dimension ({len(slice_dim)}) and the number of slicing point ({len(slice_dim)}) should be the same.'
        assert torch.all(slice_dim.to(dtype=int)==slice_dim), 'slicing dimension should be integer'

        slice_dim = slice_dim.to(dtype=int)
        slice_pt = slice_pt.to(dtype=torch.float64)

        N = len(slice_dim)
        
        Z = self.Z
        c = self.center
        G = self.generators

        slice_idx = torch.zeros(N,dtype=int)
        for i in range(N):
            non_zero_idx = np_argwhere(G[slice_dim[i],:] != 0)[0]
            # NOTE: recent version (1.10.XX) of torch doesn't provide argwhere function, but upcomming version (1.11.XX) will include this.
            if len(non_zero_idx) != 1:
                if len(non_zero_idx) == 0:
                    raise ValueError('no generator for slice index')
                else:
                    raise ValueError('more than one generator for slice index')
            slice_idx[i] = non_zero_idx

        slice_c = c[slice_dim];

        slice_G = torch.zeros(N,N,dtype=torch.float64)
        for i in range(N):
            slice_G[i] = G[slice_dim[i],slice_idx]
        
        slice_lambda = torch.linalg.solve(slice_G, slice_pt - slice_c)

        assert not any(abs(slice_lambda)>1), 'slice point is ouside bounds of reach set, and therefore is not verified'

        Z_new = torch.zeros(Z.shape)
        Z_new[:,0] = c + G[:,slice_idx]@slice_lambda
        Z_new[:,1:] = G
        Z_new = delete_column(Z_new,slice_idx+1)
        return zonotope(Z_new)

    def project(self,dim=[0,1]):
        '''
        the projection of a zonotope onto the specified dimensions
        self: <zonotope>
        dim: <int> or <list> or <np.ndarray> or <torch.Tensor> dimensions for prjection 
        
        return <zonotope>
        '''
        Z = self.Z[dim,:]
        return zonotope(Z)

    def polygon2d(self):
        '''
        converts a 2-d zonotope into a polygon as vertices
        self: <zonotope>

        return <torch.Tensor>, <torch.float64>
        '''
        dim = 2
        z = self.deleteZerosGenerators()
        c = z.center
        G = z.generators
        n = z.n_generators
        x_max = torch.sum(abs(G[0,:]))
        y_max = torch.sum(abs(G[1,:]))
        G[:,G[1,:]<0] = - G[:,G[1,:]<0] # make all y components as positive
        angles = torch.atan2(G[1,:], G[0,:]) % 2*torch.pi
        ang_idx = torch.argsort(angles)
        
        vertices_half = torch.zeros(dim,n+1)
        for i in range(n):
            vertices_half[:,i+1] = vertices_half[:,i] + 2*G[:,ang_idx[i]]
        
        vertices_half[0,:] += (x_max-torch.max(vertices_half[0,:]))
        vertices_half[1,:] -= y_max

        full_vertices = torch.zeros(dim,2*n+1)
        full_vertices[:,:n+1] = vertices_half
        full_vertices[:,n+1:] = -vertices_half[:,1:] + vertices_half[:,0].reshape(dim,1) + vertices_half[:,-1].reshape(dim,1) #flipped
        
        full_vertices += c.reshape(dim,1)
        return full_vertices.to(dtype=torch.float64)

    def deleteZerosGenerators(self):
        '''
        delete zero vector generators
        self: <zonotope>

        return <zonotope>
        '''
        G = self.generators 
        non_zero_idxs = torch.any(G!=0,axis=0).to(dtype=int)
        Z_new = torch.zeros(self.dim,sum(non_zero_idxs)+1)
        j=0
        for i in range(G.shape[1]):
            if non_zero_idxs[i]:
                j += 1
                Z_new[:,j] = G[:,i]
        Z_new[:,0] = self.center
        return zonotope(Z_new)

    def plot2d(self, ax,facecolor='green',edgecolor='green'):
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
        
        import matplotlib.patches as patches
        # dim = 2 or 3
        # Define fig = plt.figure() priori
        # and  Take ax = fig.gca() as input 
        dim = 2
        z = self.project(torch.arange(dim))
        p = z.polygon2d()

        ax.add_patch(patches.Polygon(p.T,alpha=.5,edgecolor=edgecolor,facecolor=facecolor,linewidth=2,))




if __name__ == '__main__':
    import matplotlib.pyplot as plt
        
    Z = torch.tensor([[0, 1, 0,1],[0, 0, -1,1],[0,0,0,1]])
    z = zonotope(Z)
    print(z.Z)
    print(z.slice(2,1).Z)
    print(z)

    fig = plt.figure()    
    ax = fig.gca() 
    z.plot2d(ax)

  
    Z1 = torch.tensor([[5, 1, 0,1],[5, 0, -1,1]])
    z1 = zonotope(Z1)
    z1.plot2d(ax)
    plt.axis([-5,5,-5,5])
    plt.show()

# %%
