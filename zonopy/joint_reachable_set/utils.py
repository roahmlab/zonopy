import torch
from zonopy import polyZonotope, matPolyZonotope
import numpy as np

def remove_dependence_and_compress(Set,id):
    # id_idx = Set.id == id
    id_idx = np.any(np.expand_dims(Set.id,1) == id, axis=1)

    has_val = torch.any(Set.expMat[:,id_idx] != 0, dim=1)
    dn_has_val = torch.all(Set.expMat[:,~id_idx] == 0, dim=1)
    ful_slc_idx = torch.logical_and(has_val, dn_has_val)
    # ful_slc_idx = torch.all(torch.vstack((Set.expMat[id_idx] !=0,Set.expMat[~id_idx] == 0)),dim=0)
    #print(Set.expMat)
    #print(Set.id)
    #print(id_idx)
    #print(Set.expMat[id_idx])
    #print(Set.expMat[~id_idx])
    #print(ful_slc_idx)
    assert torch.count_nonzero(ful_slc_idx) <= np.count_nonzero(id_idx)
    # assert Set.expMat.shape[1] ==0 or sum(ful_slc_idx) >= 1
    
    if isinstance(Set,polyZonotope):
        c = Set.c
        # G = Set.G[:,ful_slc_idx]\
        # Grest = torch.sum(abs(Set.G[:,~ful_slc_idx]),axis=-1) + torch.sum(abs(Set.Grest),axis=-1)
        # ExpMat = Set.expMat[id_idx,ful_slc_idx].reshape(1,-1)
        # return polyZonotope(c,G,Grest,ExpMat,id.reshape(-1),Set.dtype,Set.itype,Set.device)
        G = Set.G[ful_slc_idx]
        # Instead of reducing Grest now, just leave it
        ExpMat = Set.expMat[ful_slc_idx][:,id_idx]
        Z = torch.concat([c.unsqueeze(0), G, Set.G[~ful_slc_idx], Set.Grest], dim=0)
        return polyZonotope(Z, G.shape[0], ExpMat, Set.id[id_idx])
    elif isinstance(Set,matPolyZonotope):
        C = Set.C
        # G = Set.G[:,:,ful_slc_idx]
        # G = Set.G[ful_slc_idx,:,:]
        # Grest = torch.cat((Set.G[:,:,~ful_slc_idx],Set.Grest),dim=-1)
        # ExpMat = Set.expMat[id_idx,ful_slc_idx].reshape(1,-1)
        # return matPolyZonotope(C,G,Grest,ExpMat,id.reshape(-1),Set.dtype,Set.itype,Set.device)
        G = Set.G[ful_slc_idx]
        ExpMat = Set.expMat[ful_slc_idx][:,id_idx]
        Z = torch.concat([C.unsqueeze(0), G, Set.G[~ful_slc_idx], Set.Grest], dim=0)
        return matPolyZonotope(Z, G.shape[0], ExpMat, Set.id[id_idx])

