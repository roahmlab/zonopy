"""
Utilities for polynomial zonotope and matrix polynomial zonotope
Author: Yongseok Kwon
Reference: CORA
"""
from __future__ import annotations
import torch
from zonopy.conSet import DEFAULT_OPTS
from typing import Tuple

@torch.jit.script
def _removeRedundantExponentsScript(ExpMat: torch.Tensor, G: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    add up all generators that belong to terms with identical exponents
    
    ExpMat: <torch.tensor> matrix containing the exponent vectors
    G: <torch.tensor> generator matrix
    
    return,
    ExpMat: <torch.tensor> modified exponent matrix
    G: <torch.tensor> modified generator matrix
    '''
    # True for non-zero generators
    # NOTE: need to fix or maybe G should be zero tensor for empty

    # non-zero generator index
    idxD = torch.sum(G!=0,dim=(0,2))!=0

    if G.shape[1] == 0 or not idxD.any():
        Gnew = torch.empty(0,dtype=G.dtype,device=G.device).reshape(G.shape[0], 0, G.shape[2])
        ExpMatNew = torch.empty(0,dtype=ExpMat.dtype,device=G.device).reshape(0, ExpMat.shape[1])
        return ExpMatNew, Gnew
    elif not idxD.all():
        G = G[:,idxD,:]
        ExpMat = ExpMat[idxD]

    ExpMatNew, ind = torch.unique(ExpMat,dim=0,return_inverse=True,sorted=True)
    if ind.max()+1 == ind.numel():
        return ExpMatNew, G[:,ind.argsort(),:]

    # This uses atomicAdd which is nondeterministic, but for integers should
    # have deterministic outputs.
    Gnew = torch.zeros_like(G[:, :ExpMatNew.shape[0], :])
    Gnew.index_add_(1, ind, G)

    return ExpMatNew, Gnew

def removeRedundantExponentsBatch(ExpMat,G,batch_idx_all,dim_N=2):
    '''
    add up all generators that belong to terms with identical exponents
    
    ExpMat: <torch.tensor> matrix containing the exponent vectors
    G: <torch.tensor> generator matrix
    
    return,
    ExpMat: <torch.tensor> modified exponent matrix
    G: <torch.tensor> modified generator matrix
    '''
    # True for non-zero generators
    # NOTE: need to fix or maybe G should be zero tensor for empty
    batch_shape = G.shape[:-dim_N]
    N_shape = G.shape[-dim_N+1:]
    ExpMatNew, Gnew = _removeRedundantExponentsScript(ExpMat, G.flatten(-dim_N+1, -1).flatten(0,-3))
    return ExpMatNew, Gnew.reshape(batch_shape+(-1,)+N_shape)

def removeRedundantExponents(ExpMat,G):
    '''
    add up all generators that belong to terms with identical exponents
    
    ExpMat: <torch.tensor> matrix containing the exponent vectors
    G: <torch.tensor> generator matrix
    
    return,
    ExpMat: <torch.tensor> modified exponent matrix
    G: <torch.tensor> modified generator matrix
    '''
    # True for non-zero generators
    # NOTE: need to fix or maybe G should be zero tensor for empty
    N_shape = G.shape[1:]
    ExpMatNew, Gnew = _removeRedundantExponentsScript(ExpMat, G.flatten(1, -1).unsqueeze(0))
    return ExpMatNew, Gnew.reshape((-1,)+N_shape)

import numpy as np
def mergeExpMatrix(id1, id2, expMat1, expMat2):

    if len(id1) == len(id2) and all(id1==id2):
        return id1, expMat1, expMat2
    
    ind2 =np.zeros_like(id2)
    Ind_rep = id2.reshape(-1,1) == id1
    ind = np.any(Ind_rep,axis=1)
    non_ind = ~ind
    ind2[ind] = Ind_rep.nonzero()[1]
    ind2[non_ind] = np.arange(non_ind.sum()) + len(id1)
    id = np.hstack((id1,id2[non_ind]))
    # preallocate
    # expMat1_out = torch.zeros((len(expMat1), len(id)), dtype=torch.int64)
    # expMat2_out = torch.zeros((len(expMat2), len(id)), dtype=torch.int64)
    # # Store out
    # expMat1_out[:,:len(id1)] = expMat1
    # expMat2_out[:,ind2] = expMat2
    # return id, expMat1_out, expMat2_out
    expMat = torch.zeros((len(expMat1)+len(expMat2), len(id)), dtype=expMat1.dtype, device=expMat1.device)
    expMat[:len(expMat1),:len(id1)] = expMat1
    expMat[len(expMat1):,ind2] = expMat2
    return id, expMat[:len(expMat1)], expMat[len(expMat1):]


def mergeExpMatrix_old(id1, id2, expMat1, expMat2):
    '''
    Merge the ID-vectors of two polyZonotope and adapt the  matrice accordingly
    id1: <>,
    id2:
    expMat1: <>
    expMat1: <>
    
    return,
    id: <>, merged ID-vector
    expMat1: <>, adapted exponent matric of the first polynomial zonotope
    expMat2: <>, adapted exponent matric of the second polynomial zonotope
    '''
    L1 = len(id1)
    L2 = len(id2)

    import numpy as np
    # ID vectors are identical
    if L1 == L2 and all(id1==id2):
        id = id1
        return id, expMat1, expMat2

    elif isinstance(id1, np.ndarray):
        ind2 =np.zeros_like(id2)

        Ind_rep = id2.reshape(-1,1) == id1
        ind = np.any(Ind_rep,axis=1)
        non_ind = ~ind
        ind2[ind] = Ind_rep.nonzero()[1]
        ind2[non_ind] = np.arange(non_ind.sum()) + len(id1)
        id = np.hstack((id1,id2[non_ind]))

    # ID vectors not identical -> MERGE
    else:
        ind2 =torch.zeros_like(id2)

        Ind_rep = id2.reshape(-1,1) == id1
        ind = torch.any(Ind_rep,axis=1)
        non_ind = ~ind
        ind2[ind] = Ind_rep.nonzero()[:,1]
        ind2[non_ind] = torch.arange(non_ind.sum(),device=non_ind.device) + len(id1)
        id = torch.hstack((id1,id2[non_ind]))
    
    # construct the new exponent matrices
    L = len(id)
    expMat1 = torch.hstack((expMat1,torch.zeros(len(expMat1),L-L1,dtype=expMat1.dtype,device=expMat1.device)))
    temp = torch.zeros(len(expMat2),L,dtype=expMat1.dtype,device=expMat1.device)
    temp[:,ind2] = expMat2
    expMat2 = temp

    return id, expMat1, expMat2

def pz_repr(pz):
    # TODO: number of show 
    # TODO: precision

    pz_mat = pz.Z.T
    pz_repr = 'polyZonotope('
    for i in range(pz.dimension):
        if i == 0 and pz.dimension>1:
            pz_repr += '[['
        elif i ==0:
            pz_repr += '['
        else:
            pz_repr += '              ['
        for j in range(pz_mat.shape[1]):
            pz_repr +=  ('{{:.{}f}}').format(4).format(pz_mat[i,j].item())
            
            if j == 0 or (j != pz_mat.shape[1]-1 and j == pz.G.shape[0]):
                pz_repr += ' | '
            elif j !=pz_mat.shape[1]-1:
                pz_repr +=', '
        if i != pz_mat.shape[0]-1:
            pz_repr +='],\n'
        elif pz_mat.shape[0]>1:
            pz_repr += ']]'
        else:
            pz_repr += ']'

    if pz.dtype != DEFAULT_OPTS.DTYPE:
        pz_repr += f', dtype={pz.dtype}'
 
    pz_repr += '\n       '
    
    expMat_str = str(pz.expMat.to(dtype=int,device='cpu'))

    del_dict = {'tensor':'ExpMat=','    ':'       ','(':'',')':''}
    for del_el in del_dict.keys():
        expMat_str = expMat_str.replace(del_el,del_dict[del_el])

    pz_repr += expMat_str
    if pz.itype != DEFAULT_OPTS.ITYPE:
        pz_repr += f', itype={pz.itype}'
    if pz.device != DEFAULT_OPTS.DEVICE:
        pz_repr += f', device={pz.device}'
    pz_repr += ')'
    return pz_repr
'''
def check_decimals(tensor):
    max_digits, min_digits = 1,1
    max 
    if len(tensor.shape) == 0:
        value = tensor.item()
        n_full = str(len(value))
        n_digits = str(len(round(value)))
        if n_full != n_digits:
            n_decimals = n_full-n_digits-1
        else:
            n_decimals = 0
        return n_digits, n_digits, n_decimals
    else:
        for i in range(tensor.shape[0]):
            n_digits,_,n_decimals=check_decimals(tensor[i])
    return n_digits
'''



import torch
import zonopy as zp
import numpy as np
import zonopy.internal as zpi
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union
    from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope as PZType
    from zonopy.conSet.polynomial_zonotope.batch_poly_zono import batchPolyZonotope as BPZType
    from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope as MPZType
    from zonopy.conSet.polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope as BMPZType

def remove_dependence_and_compress(
        Set: Union[PZType, BPZType, MPZType, BMPZType],
        id: np.ndarray
        ) -> Union[PZType, BPZType, MPZType, BMPZType]:
    
    # # First compress (We can address this!)
    Set.compress(2)

    # Proceed
    id = np.asarray(id, dtype=int)
    id_idx = np.any(np.expand_dims(Set.id,1) == id, axis=1)

    has_val = torch.any(Set.expMat[:,id_idx] != 0, dim=1)
    dn_has_val = torch.all(Set.expMat[:,~id_idx] == 0, dim=1)
    ful_slc_idx = torch.logical_and(has_val, dn_has_val)
    
    if isinstance(Set,(zp.polyZonotope,zp.batchPolyZonotope)):
        if zpi.__debug_extra__: assert torch.count_nonzero(ful_slc_idx) <= np.count_nonzero(id_idx)
        c = Set.c
        G = Set.G[...,ful_slc_idx,:]
        ExpMat = Set.expMat[ful_slc_idx][:,id_idx]
        # Instead of reducing Grest now, just leave it
        Z = torch.concat([c.unsqueeze(-2), G, Set.G[...,~ful_slc_idx,:], Set.Grest], dim=-2)
        return type(Set)(Z, G.shape[-2], ExpMat, Set.id[id_idx])
        
    elif isinstance(Set,(zp.matPolyZonotope,zp.batchMatPolyZonotope)):
        # TODO WHY NO ASSERT HERE
        C = Set.C
        G = Set.G[...,ful_slc_idx,:,:]
        ExpMat = Set.expMat[ful_slc_idx][:,id_idx]
        # Instead of reducing Grest now, just leave it
        Z = torch.concat([C.unsqueeze(-3), G, Set.G[...,~ful_slc_idx,:,:], Set.Grest], dim=-3)
        return type(Set)(Z, G.shape[-3], ExpMat, Set.id[id_idx])
    else:
        raise NotImplementedError


if __name__ == '__main__':
    
    expMat = torch.tensor([[1],[2],[1]])
    G = torch.tensor([[2],[3],[4]])

    expMat, G = removeRedundantExponents(expMat,G)
    print(G)
    print(expMat)

    expMat1 = torch.tensor([[1,2],[2,3],[1,4]])
    id1 = torch.tensor([1,2])
    expMat2 = torch.tensor([[1,2],[2,3],[1,4]])
    id2 = torch.tensor([2,3])
    id, expMat1,expMat2 = mergeExpMatrix(id1,id2,expMat1,expMat2)    

    print(id,expMat1,expMat2)