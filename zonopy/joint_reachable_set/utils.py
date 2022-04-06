import torch
from zonopy import polyZonotope, matPolyZonotope
def remove_dependence_and_compress(Set,id):
    id_idx = Set.id == id
    ful_slc_idx = torch.all(torch.vstack((Set.expMat[id_idx] !=0,Set.expMat[~id_idx] == 0)),dim=0)
    assert sum(ful_slc_idx) >= 1
    if isinstance(Set,polyZonotope):
        c = Set.c
        G = Set.G[:,ful_slc_idx]
        Grest = torch.sum(abs(Set.G[:,~ful_slc_idx]),axis=-1) + torch.sum(abs(Set.Grest),axis=-1)
        ExpMat = Set.expMat[id_idx,ful_slc_idx].reshape(1,-1)
        return polyZonotope(c,G,Grest,ExpMat,id.reshape(-1),Set.dtype,Set.itype,Set.device)
    elif isinstance(Set,matPolyZonotope):
        C = Set.C
        G = Set.G[:,:,ful_slc_idx]
        Grest = torch.cat((Set.G[:,:,~ful_slc_idx],Set.Grest),dim=-1)
        ExpMat = Set.expMat[id_idx,ful_slc_idx].reshape(1,-1)
        return matPolyZonotope(C,G,Grest,ExpMat,id.reshape(-1),Set.dtype,Set.itype,Set.device)

