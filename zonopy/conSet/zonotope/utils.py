"""
Utilities for zonotope and matrix zonotope
Author: Yongseok Kwon
Reference: CORA
"""

import torch

def pickedGenerators(c,G,order):
    '''
    selects generators to be reduced
    '''
    dim = c.shape
    dim_c = len(dim)
    norm_dim = tuple(range(1,dim_c+1))
    permute_order = (dim_c,) + tuple(range(dim_c))
    reverse_order = norm_dim+(0,)
    G = G.permute(permute_order)

    Gunred = torch.tensor([]).reshape(dim+(0,))
    Gred = torch.tensor([]).reshape(dim+(0,))
    if G.numel() != 0:
        d = torch.prod(torch.tensor(G.shape[1:]))
        nrOfGens = G.shape[0]
        # only reduce if zonotope order is greater than the desired order
        if nrOfGens > d*order:
            
            # compute metric of generators
            h = torch.linalg.vector_norm(G,1,norm_dim) - torch.linalg.vector_norm(G,torch.inf,norm_dim)

            # number of generators that are not reduced
            nUnreduced = int(d*(order-1))
            nReduced = nrOfGens - nUnreduced 
            # pick generators with smallest h values to be reduced
            sorted_h = torch.argsort(h)
            ind_red = sorted_h[:nReduced]
            ind_rem = sorted_h[nReduced:]
            Gred = G[ind_red].permute(reverse_order)
            # unreduced generators
            #Gunred = delete_column(G,ind_red)
            Gunred = G[ind_rem].permute(reverse_order)
        else:
            Gunred = G.permute(reverse_order)

    return c, Gunred, Gred


def ndimCross(Q):
    '''
    computes the n-dimensional cross product
    Q: (n+1) x n
    '''
    dim = len(Q)
    v = torch.zeros(dim,device=Q.device)
    indices = torch.arange(dim,device=Q.device)
    for i in range(dim):
        v[i] = (-1)**i*torch.det(Q[i != indices])
    return v

if __name__ == '__main__':
    v = ndimCross(torch.Tensor([[1,2,3],[4,5,1],[6,1,6],[3,4,5]]))

    print(v)