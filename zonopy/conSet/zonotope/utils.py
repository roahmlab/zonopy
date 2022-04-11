"""
Utilities for zonotope and matrix zonotope
Author: Yongseok Kwon
Reference: CORA
"""
from zonopy.conSet.utils import delete_column

import torch

def pickedGenerators(Z,order):
    '''
    selects generators to be reduced
    '''
    Z = Z.deleteZerosGenerators()
    c = Z.center
    G = Z.generators
    Gunred = torch.tensor([]).reshape(Z.dimension,0)
    Gred = torch.tensor([]).reshape(Z.dimension,0)
    if G.numel() != 0:
        d, nrOfGens = G.shape
        # only reduce if zonotope order is greater than the desired order
        if nrOfGens > d*order:
            
            # compute metric of generators
            h = torch.linalg.vector_norm(G,1,0) - torch.linalg.vector_norm(G,torch.inf,0)

            # number of generators that are not reduced
            nUnreduced = int(d*(order-1))
            nReduced = nrOfGens - nUnreduced 
            # pick generators with smallest h values to be reduced
            sorted_h = torch.argsort(h)
            ind_red = sorted_h[:nReduced]
            ind_rem = sorted_h[nReduced:]
            Gred = G[:,ind_red]
            # unreduced generators
            #Gunred = delete_column(G,ind_red)
            Gunred = G[:,ind_rem]
        else:
            Gunred = G

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