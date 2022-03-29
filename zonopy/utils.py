from turtle import distance
import matplotlib.pyplot as plt
import torch
from zonopy.conSet import zonotope, matZonotope, polyZonotope, matPolyZonotope

def plot_dict_polyzono(dict_pz,plot_freq=10,facecolor='none',edgecolor='green',linewidth=.2, hold_on=False, title=None, ax=None):
    L = 1.1
    max_key = max(dict_pz.keys())
    n_joints = max_key[0]+1
    n_time_steps = max_key[1]+1
    if ax is None:
        fig = plt.figure()    
        ax = fig.gca() 

    for i in range(n_joints):
        for t in range(n_time_steps):
            if t%plot_freq == 0:
                Z = dict_pz[(i,t)].to_zonotope()
                Z.plot2d(ax,facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)

    if not hold_on:
        if title is not None:
            plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis([-n_joints*L,n_joints*L,-n_joints*L,n_joints*L])
        plt.show()
    return ax

def plot_dict_zono(dict_z,plot_freq=10,facecolor='none',edgecolor='green',linewidth=.2, hold_on=False, title=None, ax=None):
    L = 1.1
    max_key = max(dict_z.keys())
    n_joints = max_key[0]+1
    n_time_steps = max_key[1]+1
    if ax is None:
        fig = plt.figure()    
        ax = fig.gca() 

    for i in range(n_joints):
        for t in range(n_time_steps):
            if t%plot_freq == 0:
                Z = dict_z[(i,t)]
                Z.plot2d(ax,facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)

    if not hold_on:
        if title is not None:
            plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis([-n_joints*L,n_joints*L,-n_joints*L,n_joints*L])
        plt.show()
    return ax


def close(zono1,zono2,eps = 1e-6):
    assert type(zono1) == type(zono2) 
    if type(zono1) == zonotope:
        assert zono1.dim == zono2.dim
        zono1, zono2 = zono1.deleteZerosGenerators(), zono2.deleteZerosGenerators()
        if zono1.n_generators != zono2.n_generators or torch.norm(zono1.center-zono2.center) > zono1.dim**(0.5)*eps:
            return False
        return compare_permuted_gen(zono1.generators,zono2.generators,eps)
    elif type(zono1) == matZonotope:
        assert zono1.n_rows == zono2.n_rows and zono1.n_cols == zono2.n_cols
        zono1, zono2 = zono1.deleteZerosGenerators(), zono2.deleteZerosGenerators()
        if zono1.n_generators != zono2.n_generators or torch.norm(zono1.center-zono2.center) > (zono1.n_rows*zono1.n_cols)**(0.5)*eps:
            return False
        return compare_permuted_gen(zono1.generators,zono2.generators,eps)
    else:
        print('Other types are not implemented yet.')
    
def compare_permuted_gen(G1, G2,eps = 1e-6):
    assert G1.shape == G2.shape

    dim_G = len(G1.shape)
    permute_order = [dim_G-1] + list(range(dim_G-1))
    reverse_order = list(range(1,dim_G))+[0]
    G1,G2 = G1.permute(permute_order), G2.permute(permute_order)
    n_gens,dims = G1.shape[0], list(G1.shape[1:])
    
    dim_mul = 1
    for d in dims:
        dim_mul *= d
    
    for i in range(n_gens):
        distance_gen = G1[i].reshape([1]+dims)-G2
        for _ in range(dim_G-1):
            distance_gen = torch.norm(distance_gen,dim=1)
        
        closest_idx = torch.argmin(distance_gen)
        if distance_gen[closest_idx] > dim_mul**(0.5)*eps:
            return False
        G2 = G2[[j for j in range(n_gens-i) if j != closest_idx]].reshape([-1]+dims)
    return True
        

def cross(zono1,zono2):
    '''
    
    '''
    if isinstance(zono2,torch.Tensor):
        assert len(zono2.shape) == 1 and zono2.shape[0] == 3
        if isinstance(zono1,torch.Tensor):
            assert len(zono1.shape) == 1 and zono1.shape[0] == 3
            return torch.cross(zono1,zono2)
        elif isinstance(zono1,polyZonotope):
            assert zono1.dimension ==3
            return cross(-zono2,zono1)

    elif type(zono2) == polyZonotope:
        assert zono2.dimension == 3
        if type(zono1) == torch.Tensor:
            assert len(zono1.shape) == 1 and zono1.shape[0] == 3
            zono1_skew_sym = torch.tensor([[0,-zono1[2],zono1[1]],[zono1[2],0,-zono1[1]],[-zono1[1],zono1[0],0]])
            dtype = zono2.dtype
            itype = zono2.itype
            device = zono2.device

        elif type(zono1) == polyZonotope:
            assert zono1.dimension ==3
            c = zono1.c
            G = zono1.G
            Grest = zono1.Grest

            dtype = zono1.dtype
            itype = zono1.itype
            device = zono1.device
            
            Z = torch.hstack((c.reshape(-1,1),G,Grest))
            Z_skew = torch.zeros(3,3,Z.shape[1])
            
            num_c_G = 1+zono1.G.shape[1]
            
            for j in range(Z.shape[1]):
                z = Z[:,j]
                Z_skew[:,:,j] = torch.tensor([[0,-z[2],z[1]],[z[2],0,-z[1]],[-z[1],z[0],0]])
            
            zono1_skew_sym = matPolyZonotope(Z_skew[:,:,0],Z_skew[:,:,1:num_c_G],Z_skew[:,:,num_c_G:],zono1.expMat,zono1.id)

        return zono1_skew_sym@zono2    
    

def dot(zono1,zono2):
    if isinstance(zono1,torch.Tensor):
        if isinstance(zono2,polyZonotope):
            assert len(zono1.shape) == 1 and zono1.shape[0] == zono2.dimension
            zono1 = zono1.to(dtype=zono2.dtype)

            c = (zono1@zono2.c).reshape(1)
            G = (zono1@zono2.G).reshape(1,-1)
            Grest = (zono1@zono2.Grest).reshape(1,-1)
            return polyZonotope(c,G,Grest,zono2.expMat,zono2.id,zono2.dtype,zono2.itype,zono2.device)

if __name__ == '__main__':
    A = torch.arange(10,dtype=float).reshape(2,5)
    #import pdb; pdb.set_trace()
    B = A[:,torch.randperm(5)]
    print(compare_permuted_gen(A,B))
    A = torch.arange(60,dtype=float).reshape(2,6,5)
    B = A[:,:,torch.randperm(5)]
    print(compare_permuted_gen(A,B))
    
    



