#%%
# TODO:
# 1. represent zonotope in polytope
# 2. represent zonotope in polygon 3D
# 3. plot 3D

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Zonotope:
    def __init__(self,Z):
        #
        #
        #
        self.Z = Z
        self.center = self.Z[:,0]
        self.generators = self.Z[:,1:]
        self.dim = self.Z.shape[0]
        self.n_generators = self.Z.shape[1] - 1

    def  __add__(self,other):   
        if type(other) == np.ndarray:
            Z = self.Z
            if other.shape != Z[:,0].shape:
                raise ValueError('array dimension does not match: should be '+str(Z[:,0].shape)+', but '+str(other.shape))
            Z[:,0] += other
                
        elif type(other) == Zonotope: # Minkowski sum
            if self.dim != other.dim:
                raise ValueError('zonotope dimension does not match: '+str(self.dim)+' and '+str(other.dim))
            
            Z = np.zeros([self.dim,1+self.n_generators+other.n_generators])
            Z[:,0] = self.center + other.center
            Z[:,1:1+self.n_generators] = self.generators
            Z[:,1+self.n_generators:] = other.generators
            # need to be reduced NOTE: I forgot what did I mean for this
        else:
            raise ValueError('the other object is neither a zonotope nor an array: '+str(type(other)))

        out = Zonotope(Z)
        return out

    def  __sub__(self,other):
        if type(other) == np.ndarray:
            Z = self.Z
            if other.shape != Z[:,0].shape:
                raise ValueError('array dimension does not match: should be '+str(Z[:,0].shape)+', but '+str(other.shape))
            Z[:,0] -= other
                
        elif type(other) == zonotope: # Minkowski sum
            if self.dim != other.dim:
                raise ValueError('zonotope dimension does not match: '+str(self.dim)+' and '+str(other.dim))
            
            Z = np.zeros([self.dim,1+self.n_generators+other.n_generators])
            Z[:,0] = self.center - other.center
            Z[:,1:1+self.n_generators] = self.generators
            Z[:,1+self.n_generators:] = other.generators
            
        else:
            raise ValueError('the other object is neither a zonotope nor an array: '+str(type(other)))

        out = Zonotope(Z)
        return out
    
    def __neg__(self):
        Z = -self.Z
        Z[:,1:] = self.Z[:,1:]
        out = Zonotope(Z)
        return out    
    
    def __pos__(self):
        return self

    def __matmul__(self,matrix):
        # NOTE: the order is less intuitive
        Z = matrix @ self.Z
        out = Zonotope(Z)
        return out  

    def slice(self,slice_dim,slice_pt):
        if type(slice_dim) != np.ndarray:
            if type(slice_dim) == list:
                slice_dim = np.array(slice_dim)
            else:
                slice_dim = np.array([slice_dim])
        if type(slice_pt) != np.ndarray:
            if type(slice_pt) == list:
                slice_pt = np.array(slice_pt)
            else:
                slice_pt = np.array([slice_pt])

        if len(slice_pt.shape) == 2 and slice_pt.shape[1] != 1:
            raise ValueError('slice point should be a column vector')
        
         
        N = len(slice_dim)
        slice_dim = slice_dim.reshape(N)
        slice_pt = slice_pt.reshape(N)

        Z = self.Z
        c = self.center
        G = self.generators

        slice_idx = np.zeros(N).astype(int)
        for i in range(N):
            non_zero_idx = np.argwhere(G[slice_dim[i],:] != 0)
            if len(non_zero_idx) != 1:
                if len(non_zero_idx) == 0:
                    raise ValueError('no generator for slice index')
                else:
                    raise ValueError('more than one generator for slice index')
            slice_idx[i] = non_zero_idx[0,0]

        slice_c = c[slice_dim];
        slice_G = G[slice_dim,slice_idx]
        if len(slice_G.shape) == 1:
            slice_G = slice_G.reshape(slice_G.shape[0],1)
        slice_lambda = LA.solve(slice_G, slice_pt - slice_c)                     

        if len(slice_lambda.shape) > 1:
            raise ValueError('slice_lambda is not 1D')
        if np.any(np.abs(slice_lambda)>1):
            raise ValueError('slice point is ouside bounds of reach set, and therefore is not verified')
        
        Z_new = np.zeros(Z.shape)
        Z_new[:,0] = c + G[:,slice_idx]@slice_lambda
        Z_new[:,1:] = G
        Z_new = np.delete(Z_new,slice_idx+1,axis=1) 
        out = Zonotope(Z_new)
        return out

    def project(self,dim=[0,1]):
        # dim: list or array
        Z = self.Z[dim,:]
        out = Zonotope(Z)
        return out

    def polygon(self, dim=2):
        # dim = 2 or 3
        # return vertices of zonotopes
        z = self.deleteZerosGenerators()
        c = z.center
        G = z.generators
        n = z.n_generators
        x_max = np.sum(np.abs(G[0,:]))
        y_max = np.sum(np.abs(G[1,:]))
        G[:,G[1,:]<0] = - G[:,G[1,:]<0] # make all y components as positive
        angles = np.arctan2(G[1,:], G[0,:]) % 2*np.pi
        ang_idx = np.argsort(angles)
        
        vertices_half = np.zeros([dim,n+1])
        for i in range(n):
            vertices_half[:,i+1] = vertices_half[:,i] + 2*G[:,ang_idx[i]]
        
        vertices_half[0,:] += (x_max-np.max(vertices_half[0,:]))
        vertices_half[1,:] -= y_max

        full_vertices = np.zeros([dim,2*n+1])
        full_vertices[:,:n+1] = vertices_half
        full_vertices[:,n+1:] = -vertices_half[:,1:] + vertices_half[:,0].reshape(dim,1) + vertices_half[:,-1].reshape(dim,1) #flipped
        
        full_vertices += c.reshape(dim,1)
        return full_vertices

    def deleteZerosGenerators(self):
        G = self.generators 
        non_zero_idxs = np.any(G!=0,axis=0).astype(int)
        Z_new = np.zeros([self.dim,np.sum(non_zero_idxs)+1])
        j=0
        for i in range(G.shape[1]):
            if non_zero_idxs[i]:
                j += 1
                Z_new[:,j] = G[:,i]
        Z_new[:,0] = self.center
        out = Zonotope(Z_new)
        return out

    def plot(self, ax, dim=2,facecolor='green',edgecolor='green'):
        # dim = 2 or 3
        # Define fig = plt.figure() priori
        # and  Take ax = fig.gca() as input 
        z = self.project(np.arange(dim))
        p = z.polygon(dim)

        ax.add_patch(patches.Polygon(p.T,alpha=.5,edgecolor=edgecolor,facecolor=facecolor,linewidth=2,))

        return None

    def polytope(self):
        return 0




if __name__ == '__main__':
    Z = np.array([[0, 1, 0,1],[0, 0, -1,1],[0,0,0,1]])
    z = Zonotope(Z)
    print(z.Z)
    print(z.slice(2,1).Z)
    

    fig = plt.figure()    
    ax = fig.gca() 
    z.plot(ax)

  
    Z1 = np.array([[5, 1, 0,1],[5, 0, -1,1]])
    z1 = Zonotope(Z1)
    z1.plot(ax)

    plt.axis([-5,5,-5,5])
    plt.show()
    A = np.eye(2)
    b = np.array([[1,2,4],[1,2,2]])
    print(np.delete(b,np.array([0,1])+1,axis=1))

    print('__')    
    a = np.array([[1,2,3,4],[1,2,3,5]])
    print(a.size)
    print(a[:,4:].size)
# %%
