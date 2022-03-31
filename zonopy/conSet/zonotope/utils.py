"""
Utilities for zonotope and matrix zonotope
Author: Yongseok Kwon
Reference: CORA
"""

from numpy import setdiff1d
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
            ind = torch.argsort(h)[:nReduced]
            Gred = G[:,ind]
            # unreduced generators
            Gunred = delete_column(G,ind)
        else:
            Gunred = G

    return c, Gunred, Gred